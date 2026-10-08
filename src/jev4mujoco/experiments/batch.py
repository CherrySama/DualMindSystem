"""顺序启动独立的 JEV 单回合进程，保存同一版本批次的清单和统计。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from jev4mujoco.experiments.registry import EXPERIMENTS_V1
from jev4mujoco.experiments.statistics import (
    EpisodeStatisticsV1,
    aggregate_episode_statistics_v1,
    collect_episode_statistics_v1,
)
from jev4mujoco.policies.jev import JEV_MODEL


@dataclass(frozen=True, slots=True)
class BatchEpisodeV1:
    task: str
    episode: int
    relative_directory: str


def project_fingerprint_v1(root: Path) -> dict:
    """本项目没有 Git 元数据；以源码、配置及场景内容哈希标识实际版本。"""
    files = {}
    for directory in ("src", "scripts", "configs", "scenes"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".xml"):
                files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_summary(directory: Path, records: list[EpisodeStatisticsV1], planned: int) -> dict:
    per_task = {
        task: aggregate_episode_statistics_v1([record for record in records if record.task == task])
        for task in dict.fromkeys(record.task for record in records)
    }
    totals = aggregate_episode_statistics_v1(records)
    summary = {"planned_episodes": planned, "finished_episodes": len(records),
               "policy": "capability_jev_v1", "totals": totals, "per_task": per_task,
               "episodes": [asdict(record) for record in records]}
    _write_json(directory / "summary.json", summary)
    if records:
        rows = [asdict(record) for record in records]
        with (directory / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, tuple)) else value
                                 for key, value in row.items()})
    lines = ["# JEV 任务批量记录", "",
             f"已结束 {len(records)}/{planned} 回合；双重验收成功 {totals['successes']}，"
             f"首次成功 {totals['direct_successes']}，经恢复成功 {totals['recovered_successes']}，"
             f"运行错误 {totals['run_errors']}。", "",
             "每个回合对应一段视频，内部重试不另计回合。固定场景的一次运行仅说明本次结果。", "",
             "| 实验 | 轮次 | Runtime | 物理验收 | 重试请求/实际重启 | 决策 | 视频与事件 |",
             "| --- | ---: | --- | --- | ---: | ---: | --- |"]
    def verdict(value: bool | None) -> str:
        return "未知" if value is None else "通过" if value else "未通过"
    for record in records:
        video = f"[视频]({record.video_path})" if record.video_path else "视频缺失"
        lines.append(f"| {record.task} | {record.episode} | {verdict(record.runtime_success)} | "
                     f"{verdict(record.hidden_success)} | {record.retry_pick_requests + record.retry_capability_requests}/{record.actual_retry_count} | "
                     f"{record.decision_count if record.decision_count is not None else '未知'} | "
                     f"{video} · [事件]({record.events_path}) · [统计]({record.run_directory}/statistics.json) |")
    lines.extend(["", "实际重试计数来自 Runtime 的 recovery_reset；抓取链、推动和按压均计入每任务的共同额度。"
                  "仅观察和被拒绝的恢复请求不计入实际重试。物理失败、验收分歧、决策耗尽和接口异常可能重叠，不能相加为失败总数。", ""])
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_batch_v1(
    root: Path,
    tasks: tuple[str, ...],
    episodes: int,
    *,
    success_tail_seconds: float = 2.0,
    stack_positions_json: Path | None = None,
    maximum_retries: int = 5,
    jev_stages: int = 1,
) -> Path:
    if episodes <= 0 or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("batch requires unique tasks and positive episode count")
    if type(maximum_retries) is not int or maximum_retries < 0:
        raise ValueError("maximum_retries must be nonnegative")
    if type(jev_stages) is not int or jev_stages not in (1, 2):
        raise ValueError("jev_stages must be 1 or 2")
    if any(task not in EXPERIMENTS_V1 for task in tasks):
        raise ValueError("batch contains an unknown experiment")
    if stack_positions_json is not None and tasks != ("stack",):
        raise ValueError("custom stack positions require a stack-only batch")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / "runs" / "batches" / f"jev_{timestamp}"
    directory.mkdir(parents=True)
    if (root / "configs").is_dir():
        shutil.copytree(root / "configs", directory / "configs")
        shutil.copytree(root / "scenes", directory / "scenes")
    plan = [BatchEpisodeV1(task, episode, f"{task}/episode_{episode:03d}")
            for task in tasks for episode in range(1, episodes + 1)]
    positions_copy = None
    if stack_positions_json is not None:
        positions_copy = directory / "stack_positions.json"
        positions_copy.write_bytes(stack_positions_json.read_bytes())
    fingerprint = project_fingerprint_v1(root)
    manifest = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "capability_jev_v1", "jev_model": JEV_MODEL,
        "python": sys.version, "python_executable": sys.executable,
        "dependencies": {name: version(name) for name in ("mujoco", "numpy", "httpx", "imageio", "imageio-ffmpeg")},
        "tasks": list(tasks), "episodes_per_task": episodes,
        "success_tail_seconds": success_tail_seconds,
        "maximum_retries_per_task": maximum_retries,
        "jev_stages": jev_stages,
        "stack_positions_json": str(positions_copy) if positions_copy else None,
        "fingerprint": fingerprint,
        "plan": [{**asdict(item), "maximum_decisions": EXPERIMENTS_V1[item.task].maximum_decisions,
                  "runtime_kwargs": EXPERIMENTS_V1[item.task].runtime_kwargs} for item in plan],
    }
    _write_json(directory / "manifest.json", manifest)
    records: list[EpisodeStatisticsV1] = []
    _save_summary(directory, records, len(plan))
    print(f"批次目录：{directory}", flush=True)
    for index, item in enumerate(plan, start=1):
        if project_fingerprint_v1(root)["sha256"] != fingerprint["sha256"]:
            manifest["status"] = "source_changed"
            _write_json(directory / "manifest.json", manifest)
            raise RuntimeError("source/config/scene changed during batch; stopped to preserve version consistency")
        episode_directory = directory / item.relative_directory
        episode_directory.mkdir(parents=True)
        command = [sys.executable, str(root / "scripts/run_v1.py"), "--task", item.task,
                   "--output-directory", str(episode_directory),
                   "--success-tail-seconds", str(success_tail_seconds),
                   "--maximum-retries", str(maximum_retries), "--jev-stages", str(jev_stages)]
        if positions_copy is not None:
            command.extend(["--stack-positions-json", str(positions_copy)])
        print(f"[{index}/{len(plan)}] 开始 {item.task}，第 {item.episode} 回合", flush=True)
        started = time.perf_counter()
        with (episode_directory / "process.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        record = collect_episode_statistics_v1(item.task, item.episode, episode_directory,
                                               process.returncode, time.perf_counter() - started)
        records.append(record)
        _write_json(episode_directory / "statistics.json", asdict(record))
        _save_summary(directory, records, len(plan))
        print(f"[{index}/{len(plan)}] {item.task}："
              f"{'运行错误' if record.run_error else '成功' if record.overall_success else '未通过'}，"
              f"重试请求 {record.retry_pick_requests + record.retry_capability_requests}，实际重启 {record.actual_retry_count}；{record.reason}", flush=True)
    manifest["status"] = "completed"
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["final_fingerprint"] = project_fingerprint_v1(root)
    manifest["source_unchanged"] = manifest["final_fingerprint"]["sha256"] == fingerprint["sha256"]
    _write_json(directory / "manifest.json", manifest)
    print(f"批次统计：{directory / 'summary.md'}", flush=True)
    return directory
