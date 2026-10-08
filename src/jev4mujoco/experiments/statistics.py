"""按同一回合的结果与事件统计；回合内恢复不增加样本数量。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EpisodeStatisticsV1:
    task: str
    episode: int
    run_directory: str
    return_code: int
    elapsed_s: float
    run_error: str | None
    runtime_success: bool | None
    hidden_success: bool | None
    overall_success: bool | None
    reason: str
    success_group: str | None
    decision_count: int | None
    observation_count: int | None
    recovery_decisions: int
    observation_recovery_requests: int
    retry_pick_requests: int
    retry_capability_requests: int
    actual_retry_count: int
    actual_regrasp_count: int
    rejected_recovery_count: int
    verification_disagreement: bool | None
    decision_budget_exhausted: bool
    retry_budget_exhausted: bool
    execution_error_count: int
    interface_errors: tuple[str, ...]
    action_counts: dict[str, int]
    recovery_events: tuple[dict[str, object], ...]
    video_path: str | None
    result_path: str
    events_path: str
    log_path: str


def collect_episode_statistics_v1(
    task: str,
    episode: int,
    directory: Path,
    return_code: int,
    elapsed_s: float,
) -> EpisodeStatisticsV1:
    """退出码 1 可以是正常任务失败；没有有效双重验收结果才记为运行错误。"""
    result_path = directory / "result.json"
    events_path = directory / "events.jsonl"
    result: dict = {}
    events: list[dict] = []
    errors: list[str] = []
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("result must be an object")
    except (OSError, ValueError) as error:
        errors.append(f"invalid result: {error}")
        result = {}
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event must be an object")
            events.append(event)
    except (OSError, ValueError) as error:
        errors.append(f"invalid events: {error}")

    runtime = result.get("runtime_verifier_success")
    hidden = result.get("hidden_ground_truth", {})
    hidden = hidden.get("success") if isinstance(hidden, dict) else None
    overall = result.get("overall_success")
    decisions = result.get("decision_count")
    observations = result.get("observation_count")
    if any(type(value) is not bool for value in (runtime, hidden, overall)):
        errors.append("missing boolean runtime/hidden/overall verdicts")
    elif overall != (runtime and hidden):
        errors.append("overall verdict contradicts runtime and hidden verdicts")
    if any(type(value) is not int or value < 0 for value in (decisions, observations)):
        errors.append("missing valid decision/observation counts")
    if result.get("policy") != "capability_jev_v1":
        errors.append("result is not from the JEV policy")
    if return_code not in (0, 1) or (
        type(overall) is bool and return_code != (0 if overall else 1)
    ):
        errors.append(f"unexpected process return code: {return_code}")
    reason = str(result.get("reason", "no valid episode result"))
    if result.get("error"):
        errors.append(str(result["error"]))

    recovery_decisions = observe_requests = retry_requests = capability_requests = resets = regrasp_resets = rejected = 0
    execution_errors = 0
    interface_errors: list[str] = []
    actions: Counter[str] = Counter()
    recovery_events: list[dict[str, object]] = []
    simulation_time_s = None
    for index, event in enumerate(events, start=1):
        kind = event.get("kind")
        payload = event.get("payload")
        data = payload if isinstance(payload, dict) else {}
        if kind == "state_snapshot":
            simulation_time_s = data.get("timestamp_s")
        if kind == "action_result":
            actions[str(data.get("action", "unknown"))] += 1
            execution_errors += data.get("status") in ("failed", "rejected", "timed_out")
            simulation_time_s = data.get("end_simulation_time_s", simulation_time_s)
        if kind == "jev_exchange" and data.get("error"):
            interface_errors.append(str(data["error"]))
        is_reset = kind == "capability_runtime" and data.get("event_kind") == "recovery_reset"
        if kind == "recovery_decision":
            recovery_decisions += 1
            observe_requests += data.get("choice") == "observe"
            retry_requests += data.get("choice") == "retry_pick"
            capability_requests += data.get("choice") == "retry_capability"
        resets += is_reset
        regrasp_resets += is_reset and data.get("recovery_kind") in (None, "pick")
        rejected += kind == "recovery_rejected"
        if kind in ("recovery_decision", "recovery_rejected", "recovery_trigger", "recovery_stopped") or is_reset:
            recovery_events.append({
                "event_line": index,
                "simulation_time_s": simulation_time_s,
                "kind": "recovery_reset" if is_reset else kind,
                "choice": data.get("choice"),
                "state_id": data.get("based_on_state_id"),
                "action_epoch": data.get("based_on_action_epoch"),
                "reason": data.get("reason", payload if isinstance(payload, str) else None),
            })
    if "actual_retry_count" in result and result["actual_retry_count"] != resets:
        errors.append("result retry count contradicts actual recovery_reset events")
    if "maximum_retries" in result and (
        type(result["maximum_retries"]) is not int or result["maximum_retries"] < 0
        or resets > result["maximum_retries"]
    ):
        errors.append("actual retries exceed or invalidate configured retry budget")
    if errors:
        runtime = hidden = overall = None
        decisions = observations = None
    group = None
    if overall:
        group = ("direct" if not recovery_decisions else
                 "regrasp_recovery" if regrasp_resets else "recovery_without_regrasp")
    video_value = result.get("video_path")
    video_path = str(video_value) if isinstance(video_value, str) else None
    return EpisodeStatisticsV1(
        task=task, episode=episode, run_directory=str(directory),
        return_code=return_code, elapsed_s=elapsed_s,
        run_error="; ".join(errors) if errors else None,
        runtime_success=runtime, hidden_success=hidden, overall_success=overall,
        reason=reason, success_group=group,
        decision_count=decisions, observation_count=observations,
        recovery_decisions=recovery_decisions,
        observation_recovery_requests=observe_requests, retry_pick_requests=retry_requests,
        retry_capability_requests=capability_requests, actual_retry_count=resets,
        actual_regrasp_count=regrasp_resets, rejected_recovery_count=rejected,
        verification_disagreement=None if errors else runtime != hidden,
        decision_budget_exhausted="maximum decision budget exhausted" in reason,
        retry_budget_exhausted="maximum retry budget exhausted" in reason,
        execution_error_count=execution_errors, interface_errors=tuple(interface_errors),
        action_counts=dict(actions), recovery_events=tuple(recovery_events),
        video_path=video_path, result_path=str(result_path),
        events_path=str(events_path), log_path=str(directory / "process.log"),
    )


def aggregate_episode_statistics_v1(records: list[EpisodeStatisticsV1]) -> dict:
    """成功分组互斥；验收分歧、超时和接口错误属于可重叠诊断。"""
    count = len(records)
    success = sum(record.overall_success is True for record in records)
    groups = Counter(record.success_group for record in records if record.overall_success)
    return {
        "episodes": count,
        "valid_results": sum(record.run_error is None for record in records),
        "run_errors": sum(record.run_error is not None for record in records),
        "successes": success,
        "success_rate": success / count if count else None,
        "direct_successes": groups["direct"],
        "recovered_successes": groups["regrasp_recovery"] + groups["recovery_without_regrasp"],
        "regrasp_recovered_successes": groups["regrasp_recovery"],
        "recovered_without_regrasp_successes": groups["recovery_without_regrasp"],
        "physical_verification_failures": sum(record.hidden_success is False for record in records),
        "verification_disagreements": sum(record.verification_disagreement is True for record in records),
        "decision_budget_exhausted": sum(record.decision_budget_exhausted for record in records),
        "retry_budget_exhausted": sum(record.retry_budget_exhausted for record in records),
        "retry_pick_requests": sum(record.retry_pick_requests for record in records),
        "actual_regrasp_count": sum(record.actual_regrasp_count for record in records),
        "actual_retry_count": sum(record.actual_retry_count for record in records),
        "retry_capability_requests": sum(record.retry_capability_requests for record in records),
        "rejected_recovery_count": sum(record.rejected_recovery_count for record in records),
        "episodes_with_interface_errors": sum(bool(record.interface_errors) for record in records),
        "episodes_with_execution_errors": sum(record.execution_error_count > 0 for record in records),
        "elapsed_s": sum(record.elapsed_s for record in records),
    }
