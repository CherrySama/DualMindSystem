"""模型选择的局部意图；定义与执行记录分离，均不可变。"""

from dataclasses import dataclass
from math import isfinite
from jev4mujoco.contracts.tool_geometry import ContactSideV1


@dataclass(frozen=True, slots=True)
class LocalIntentCandidateV1:
    choice_id: str
    objective: str
    allowed_actions: tuple[str, ...]
    metric_kind: str
    target_robot_base_m: tuple[float, float, float] | None = None
    tolerance_m: float | None = None
    reference_ref: str | None = None
    maintain: tuple[str, ...] = ()
    geometry_source: str = "phase_contract"
    path_verified: bool = False
    contact_side: ContactSideV1 | None = None

    def __post_init__(self):
        if not self.choice_id.strip() or not self.objective.strip() or not self.allowed_actions:
            raise ValueError("local intent requires an identity, objective and allowed actions")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("local intent actions must be unique")
        if self.metric_kind not in ("tcp_xy", "tcp_z", "phase_contract", "contact_geometry"):
            raise ValueError("unknown local intent metric")
        if self.metric_kind == "contact_geometry" and self.contact_side is None:
            raise ValueError("contact intent requires a bound side")
        if self.metric_kind in ("tcp_xy", "tcp_z"):
            if (self.target_robot_base_m is None or len(self.target_robot_base_m) != 3
                or not all(isfinite(v) for v in self.target_robot_base_m)
                or self.tolerance_m is None or not isfinite(self.tolerance_m) or self.tolerance_m <= 0):
                raise ValueError("geometric intent requires a finite target and positive tolerance")


@dataclass(frozen=True, slots=True)
class LocalIntentChoiceV1:
    choice_id: str
    based_on_state_id: int
    based_on_action_epoch: int
    skill_instance_id: str
    phase: str

    def __post_init__(self):
        if (not self.choice_id.strip() or not self.skill_instance_id.strip() or not self.phase.strip()
            or type(self.based_on_state_id) is not int or self.based_on_state_id <= 0
            or type(self.based_on_action_epoch) is not int or self.based_on_action_epoch < 0):
            raise ValueError("local intent choice requires valid snapshot bindings")


@dataclass(frozen=True, slots=True)
class LocalIntentExecutionV1:
    intent_id: int
    candidate: LocalIntentCandidateV1
    selected_state_id: int
    selected_action_epoch: int
    status: str
    reason: str
    metric_m: float | None
    best_metric_m: float | None
    nonprogress_actions: int
    last_observed_action_epoch: int

    def __post_init__(self):
        if self.status not in ("active", "completed", "invalidated", "superseded", "cancelled"):
            raise ValueError("invalid local intent execution status")
        if (self.intent_id <= 0 or self.selected_state_id <= 0 or self.selected_action_epoch < 0
            or self.nonprogress_actions < 0 or self.last_observed_action_epoch < self.selected_action_epoch):
            raise ValueError("invalid local intent execution counters")
        for value in (self.metric_m, self.best_metric_m):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("local intent metric must be nonnegative or unknown")


@dataclass(frozen=True, slots=True)
class LocalIntentTerminationV1:
    """已结束意图的历史；不代表下一阶段的活动命令。"""
    execution: LocalIntentExecutionV1
    skill_instance_id: str
    phase: str
    based_on_state_id: int
    based_on_action_epoch: int
    based_on_observation_id: int
    trigger: str

    def __post_init__(self):
        if (self.execution.status == "active" or not self.skill_instance_id or not self.phase
            or self.trigger not in ("phase_passed", "phase_failed", "scope_changed", "run_stopped")
            or self.based_on_state_id < self.execution.selected_state_id
            or self.based_on_action_epoch < self.execution.last_observed_action_epoch
            or self.based_on_observation_id <= 0):
            raise ValueError("intent termination requires a terminal execution and current evidence binding")


@dataclass(frozen=True, slots=True)
class LocalIntentHandoffV1:
    """新观测中核验的预接触 XY 状态；不是路线历史或实际接触证明。"""
    candidate: LocalIntentCandidateV1
    skill_instance_id: str
    source_phase: str
    destination_phase: str
    based_on_state_id: int
    based_on_action_epoch: int
    based_on_observation_id: int
    metric_m: float
    condition: str = "precontact_xy_reached"

    def __post_init__(self):
        if (not self.skill_instance_id or not self.source_phase or not self.destination_phase
            or self.based_on_state_id <= 0 or self.based_on_action_epoch < 0
            or self.based_on_observation_id <= 0 or self.condition != "precontact_xy_reached"
            or self.candidate.metric_kind != "tcp_xy" or self.candidate.contact_side is None
            or self.candidate.reference_ref != self.candidate.contact_side.subject_ref
            or not isfinite(self.metric_m) or not 0 <= self.metric_m <= self.candidate.tolerance_m):
            raise ValueError("handoff requires a reached, bound precontact condition")


@dataclass(frozen=True, slots=True)
class LocalIntentContextV1:
    based_on_state_id: int
    based_on_action_epoch: int
    based_on_observation_id: int
    skill_instance_id: str
    phase: str
    candidates: tuple[LocalIntentCandidateV1, ...]
    execution: LocalIntentExecutionV1 | None = None
    replan_count: int = 0
    blocked_reason: str | None = None
    retained_approach: LocalIntentCandidateV1 | None = None
    handoff: LocalIntentHandoffV1 | None = None
    last_termination: LocalIntentTerminationV1 | None = None

    def __post_init__(self):
        if len({item.choice_id for item in self.candidates}) != len(self.candidates):
            raise ValueError("local intent candidate identities must be unique")
        if (self.based_on_state_id <= 0 or self.based_on_action_epoch < 0
            or self.based_on_observation_id <= 0 or self.replan_count < 0):
            raise ValueError("invalid local intent context binding")
        if self.execution is not None and (
            self.execution.selected_state_id > self.based_on_state_id
            or self.execution.last_observed_action_epoch > self.based_on_action_epoch):
            raise ValueError("local intent execution cannot originate from a future snapshot")
        if self.handoff is not None and (
            self.handoff.skill_instance_id != self.skill_instance_id
            or self.handoff.destination_phase != self.phase
            or self.handoff.based_on_state_id > self.based_on_state_id
            or self.handoff.based_on_action_epoch > self.based_on_action_epoch
            or self.handoff.based_on_observation_id > self.based_on_observation_id):
            raise ValueError("handoff must belong to the current scope and precede its observation")
        if self.last_termination is not None and (
            self.last_termination.based_on_state_id > self.based_on_state_id
            or self.last_termination.based_on_action_epoch > self.based_on_action_epoch
            or self.last_termination.based_on_observation_id > self.based_on_observation_id):
            raise ValueError("termination history cannot originate from a future snapshot")
