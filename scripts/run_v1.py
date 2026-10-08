from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from jev4mujoco.experiments.registry import EXPERIMENTS_V1
from jev4mujoco.experiments.batch import run_batch_v1
from jev4mujoco.policies.credentials import TYPESAFE_API_KEY_ENV, load_typesafe_api_key
from jev4mujoco.policies.jev import CapabilityJevPolicyV1
from jev4mujoco.policies.typesafe_client import TypeSafeClient


def serializable(value):
    if is_dataclass(value):
        return {key: serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def next_attempt_directory(run_name: str) -> Path:
    root = PROJECT_ROOT / f"runs/{run_name}"
    root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in root.glob("attempt_*"):
        try:
            indices.append(int(path.name.removeprefix("attempt_")))
        except ValueError:
            continue
    result = root / f"attempt_{max(indices, default=0) + 1:03d}"
    result.mkdir()
    return result


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(serializable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    task_selection = parser.add_mutually_exclusive_group()
    task_selection.add_argument(
        "--task",
        choices=tuple(EXPERIMENTS_V1),
        default="pick_and_place",
    )
    task_selection.add_argument("--tasks", nargs="+", choices=("all", *EXPERIMENTS_V1))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--maximum-retries", type=int, default=5)
    parser.add_argument("--jev-stages", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--success-tail-seconds", type=float, default=2.0)
    parser.add_argument("--stack-positions-json", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.maximum_retries < 0:
        parser.error("--maximum-retries must be nonnegative")
    if not math.isfinite(args.success_tail_seconds) or args.success_tail_seconds < 0:
        parser.error("--success-tail-seconds must be finite and nonnegative")
    tasks = tuple(args.tasks or (args.task,))
    if "all" in tasks:
        if tasks != ("all",):
            parser.error("--tasks all cannot be combined with other tasks")
        tasks = tuple(EXPERIMENTS_V1)
    if len(set(tasks)) != len(tasks):
        parser.error("--tasks must not contain duplicate experiments")
    stack_positions = None
    if args.stack_positions_json is not None:
        if tasks != ("stack",):
            parser.error("--stack-positions-json requires --task stack")
        position_data = json.loads(args.stack_positions_json.read_text(encoding="utf-8"))
        if set(position_data) != {"block_1", "block_2"}:
            parser.error("stack positions must contain block_1 and block_2")
        stack_positions = {
            name: tuple(float(value) for value in position_data[name])
            for name in ("block_1", "block_2")
        }
        if any(len(position) != 3 for position in stack_positions.values()):
            parser.error("stack positions must be XYZ triples")
    if args.tasks is not None or args.episodes > 1:
        if args.output_directory is not None:
            parser.error("--output-directory is only for a single episode")
        if not load_typesafe_api_key(PROJECT_ROOT / ".env"):
            parser.error("TYPESAFE_API_KEY is unavailable; no API request was sent")
        directory = run_batch_v1(
            PROJECT_ROOT, tasks, args.episodes,
            success_tail_seconds=args.success_tail_seconds,
            stack_positions_json=args.stack_positions_json,
            maximum_retries=args.maximum_retries,
            jev_stages=args.jev_stages,
        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["totals"]["successes"] != summary["planned_episodes"]:
            raise SystemExit(1)
        return
    experiment = EXPERIMENTS_V1[args.task]
    runtime_type = experiment.runtime_type
    run_name = experiment.run_name
    maximum_decisions = experiment.maximum_decisions
    run_name = f"{run_name}_jev"
    if args.output_directory is None:
        output_directory = next_attempt_directory(run_name)
    else:
        output_directory = args.output_directory.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        if any(path.name != "process.log" for path in output_directory.iterdir()):
            parser.error("--output-directory must contain no previous episode artifacts")
    events_path = output_directory / "events.jsonl"
    result_path = output_directory / "result.json"
    pending_video_path = output_directory / "episode.pending.mp4"
    final_video_path: Path | None = None
    video_fps = 20.0
    print(f"run_directory={output_directory}", flush=True)

    with events_path.open("w", encoding="utf-8", buffering=1) as events_handle:
        def event_sink(event: object) -> None:
            events_handle.write(
                json.dumps(serializable(event), ensure_ascii=False, sort_keys=True)
                + "\n"
            )

        writer = None
        runtime = None
        jev_client = None
        outcome = None
        failure = None
        try:
            writer = imageio.get_writer(
                pending_video_path,
                fps=video_fps,
                codec="libx264",
                macro_block_size=16,
                ffmpeg_log_level="warning",
            )
            runtime_kwargs = {
                "frame_sink": writer.append_data,
                "video_fps": video_fps,
                "event_sink": event_sink,
            }
            runtime_kwargs.update(experiment.runtime_kwargs)
            if stack_positions is not None:
                runtime_kwargs["object_initial_positions_world_m"] = stack_positions
            runtime = runtime_type(PROJECT_ROOT, **runtime_kwargs)
            write_json(output_directory / "task_plan.json", runtime.task_plan)
            write_json(output_directory / "capability_program.json", runtime.program)
            if not load_typesafe_api_key(PROJECT_ROOT / ".env"):
                raise RuntimeError(
                    "TYPESAFE_API_KEY is unavailable; no API request was sent"
                )
            jev_client = TypeSafeClient(os.environ[TYPESAFE_API_KEY_ENV])
            runtime.install_policy(
                CapabilityJevPolicyV1(
                    jev_client,
                    exchange_sink=event_sink,
                    translation_step_m=runtime.config.cartesian_step_m,
                    grasp_xy_tolerance_m=runtime.grasp_xy_tolerance_m,
                    jev_stages=args.jev_stages,
                )
            )
            outcome = runtime.run(maximum_decisions=maximum_decisions, maximum_retries=args.maximum_retries)
            write_json(output_directory / "capability_program.json", runtime.program)
            if outcome.overall_success:
                runtime.record_idle_tail(args.success_tail_seconds)
        except Exception as error:
            failure = {
                "overall_success": False,
                "reason": "unhandled_runtime_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "task": experiment.result_task,
                "policy": "capability_jev_v1",
                "run_directory": str(output_directory),
            }
        finally:
            for resource in (runtime, jev_client, writer):
                if resource is None:
                    continue
                try:
                    resource.close()
                except Exception as error:
                    if failure is None:
                        failure = {
                            "overall_success": False, "reason": "unhandled_runtime_error",
                            "error_type": type(error).__name__, "error": str(error),
                            "task": experiment.result_task, "policy": "capability_jev_v1",
                            "run_directory": str(output_directory),
                        }

    if failure is not None:
        if pending_video_path.is_file() and pending_video_path.stat().st_size:
            final_video_path = output_directory / "error_episode.mp4"
            pending_video_path.replace(final_video_path)
        failure["video_path"] = str(final_video_path) if final_video_path else None
        write_json(result_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(1)

    if outcome.overall_success:
        final_video_path = output_directory / "success_episode.mp4"
    else:
        final_video_path = output_directory / "failed_episode.mp4"
    pending_video_path.replace(final_video_path)

    result = {
        "task": experiment.result_task,
        "policy": "capability_jev_v1",
        "initial_positions_world_m": stack_positions,
        "jev_stages": args.jev_stages,
        "completed": outcome.completed,
        "blocked": outcome.blocked,
        "reason": outcome.reason,
        "decision_count": outcome.decision_count,
        "maximum_retries": args.maximum_retries,
        "actual_retry_count": runtime._retry_count,
        "maximum_decisions_per_attempt": maximum_decisions,
        "observation_count": outcome.observation_count,
        "runtime_verifier_success": outcome.runtime_verifier_success,
        "hidden_ground_truth": outcome.hidden_ground_truth,
        "overall_success": outcome.overall_success,
        "video_path": str(final_video_path),
        "video_fps": video_fps,
        "success_tail_seconds": args.success_tail_seconds,
        "final_state": outcome.final_state,
        "final_raw_state": outcome.final_raw_state,
        "run_directory": str(output_directory),
    }
    write_json(result_path, result)
    write_json(output_directory / "task_plan.json", runtime.task_plan)
    write_json(output_directory / "capability_program.json", runtime.program)
    print(json.dumps(serializable(result), ensure_ascii=False, indent=2), flush=True)
    if not outcome.overall_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
