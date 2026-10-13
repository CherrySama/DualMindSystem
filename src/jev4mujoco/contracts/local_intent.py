"""模型选择的局部意图；定义与执行记录分离，均不可变。"""

from dataclasses import dataclass
from math import hypot, isfinite
from jev4mujoco.contracts.tool_geometry import ContactSideV1


@dataclass(frozen=True, slots=True)
class PushContactPlanV1:
    """JEV历史侧面选择及能力绑定；不是当前接触或持久CAD面证明。"""
    skill_instance_id: str
    subject_ref: str
    source_ref: str | None
    target_ref: str | None
    goal: tuple[str, str, bool] | None
    selected_side: ContactSideV1
    selected_state_id: int
    selected_action_epoch: int
    selected_observation_id: int

    def __post_init__(self):
        if (not self.skill_instance_id or self.subject_ref != self.selected_side.subject_ref
            or self.selected_state_id <= 0 or self.selected_action_epoch < 0
            or self.selected_observation_id <= 0):
            raise ValueError("push plan requires a bound historical model selection")

    def matches(self, skill):
        goal = None if skill.goal is None else (skill.goal.predicate.value, skill.goal.reference_ref, skill.goal.desired_value)
        return (skill.skill_kind == "surface_push" and skill.phase == "push_toward_goal"
            and (self.skill_instance_id, self.subject_ref, self.source_ref, self.target_ref, self.goal)
            == (skill.skill_instance_id, skill.subject_ref, skill.source_ref, skill.target_ref, goal))


@dataclass(frozen=True, slots=True)
class PushPreparationV1:
    """JEV选择的再准备义务；步骤记录不是当前净空、对齐或接触证明。"""
    plan: PushContactPlanV1
    stage: str

    def __post_init__(self):
        if self.stage not in ("clearance", "align_side", "contact"):
            raise ValueError("unknown push preparation stage")


@dataclass(frozen=True, slots=True)
class PushExitProgressV1:
    """一个新鲜物体观测的方向分离余量；方向/区域不同的数字不可直接比较。"""
    state_id: int
    action_epoch: int
    observation_id: int
    region_ref: str
    region_boundary_xy_m: tuple[tuple[float, float], ...]
    required_clearance_m: float
    direction_xy: tuple[float, float]
    remaining_translation_m: float

    def __post_init__(self):
        if (self.state_id <= 0 or self.action_epoch < 0 or self.observation_id <= 0
            or not self.region_ref or len(self.region_boundary_xy_m) < 3
            or any(len(p) != 2 or not all(isfinite(v) for v in p) for p in self.region_boundary_xy_m)
            or len(self.direction_xy) != 2 or not all(isfinite(v) for v in self.direction_xy)
            or abs(hypot(*self.direction_xy) - 1.0) > 1e-6
            or any(not isfinite(v) or v < 0 for v in (self.required_clearance_m, self.remaining_translation_m))):
            raise ValueError("exit progress requires identified finite plane geometry")

    def same_plane(self, other):
        return (self.region_ref, self.region_boundary_xy_m, self.required_clearance_m, self.direction_xy) == (
            other.region_ref, other.region_boundary_xy_m, other.required_clearance_m, other.direction_xy)


@dataclass(frozen=True, slots=True)
class PushGoalObservationV1:
    state_id: int
    action_epoch: int
    observation_id: int
    vertex_signed_boundary_distances_m: tuple[float, ...]
    vertex_outside_distances_m: tuple[float, ...] | None
    footprints_overlap: bool | None
    footprint_clearance_m: float | None
    goal_satisfied: bool
    exit_progress: tuple[PushExitProgressV1, ...] = ()

    def __post_init__(self):
        if (self.state_id <= 0 or self.action_epoch < 0 or self.observation_id <= 0
            or len(self.vertex_signed_boundary_distances_m) < 3
            or not all(isfinite(v) for v in self.vertex_signed_boundary_distances_m)
            or type(self.goal_satisfied) is not bool):
            raise ValueError("push goal observation requires identified finite native geometry")
        outside = self.vertex_outside_distances_m
        if outside is not None and (len(outside) != len(self.vertex_signed_boundary_distances_m)
            or not all(isfinite(v) and v >= 0 for v in outside)):
            raise ValueError("outside distances must correspond to the observed vertices")
        if self.footprints_overlap is not None and type(self.footprints_overlap) is not bool:
            raise ValueError("overlap must be observed Boolean or unknown")
        if self.footprint_clearance_m is not None and (
            not isfinite(self.footprint_clearance_m) or self.footprint_clearance_m < 0):
            raise ValueError("footprint clearance must be finite and nonnegative")
        if any((p.state_id, p.action_epoch, p.observation_id) != (
            self.state_id, self.action_epoch, self.observation_id) for p in self.exit_progress):
            raise ValueError("exit progress must belong to this observed frame")


@dataclass(frozen=True, slots=True)
class PushGoalExecutionV1:
    """模型选择的目标计划；派生步骤不是新的模型选择，也不是当前接触证明。"""
    plan: PushContactPlanV1
    selected_choice_id: str
    stage: str
    status: str
    reason: str
    baseline: PushGoalObservationV1 | None = None
    current_goal: PushGoalObservationV1 | None = None
    best_goal: PushGoalObservationV1 | None = None
    actions_without_goal_progress: int = 0
    last_counted_action_epoch: int = 0
    advancement_started: bool = False
    best_exit_progress: tuple[PushExitProgressV1, ...] = ()

    def __post_init__(self):
        if (self.stage not in ("clearance", "align_side", "contact", "advance")
            or self.status not in ("active", "suspended", "invalidated", "completed", "cancelled")
            or not self.selected_choice_id or self.actions_without_goal_progress < 0
            or self.last_counted_action_epoch < self.plan.selected_action_epoch):
            raise ValueError("invalid persistent push goal execution")


@dataclass(frozen=True, slots=True)
class PlacementCorrectionV1:
    """放置净空纠正的流程义务；触发/完成身份是历史，不是当前物理证明。"""
    skill_instance_id: str
    phase: str
    subject_ref: str
    source_ref: str | None
    target_ref: str
    goal: tuple[str, str, bool] | None
    trigger_state_id: int
    trigger_action_epoch: int
    trigger_observation_id: int
    status: str = "pending_clearance"
    resolved_state_id: int | None = None
    resolved_action_epoch: int | None = None
    resolved_observation_id: int | None = None

    def __post_init__(self):
        if (not self.skill_instance_id or self.phase != "lower_to_release"
            or not self.subject_ref or not self.target_ref or self.status not in ("pending_clearance", "completed")
            or self.trigger_state_id <= 0 or self.trigger_action_epoch < 0 or self.trigger_observation_id <= 0):
            raise ValueError("placement correction requires a bound trigger and status")
        resolved = (self.resolved_state_id, self.resolved_action_epoch, self.resolved_observation_id)
        if self.status == "completed":
            if any(v is None or v < first for v, first in zip(resolved,
                (self.trigger_state_id, self.trigger_action_epoch, self.trigger_observation_id))):
                raise ValueError("correction completion must follow its trigger")
        elif any(v is not None for v in resolved):
            raise ValueError("pending correction cannot carry completion evidence")

    def matches(self, skill):
        goal = None if skill.goal is None else (skill.goal.predicate.value, skill.goal.reference_ref, skill.goal.desired_value)
        return skill.skill_kind == "place" and (self.skill_instance_id, self.phase, self.subject_ref,
            self.source_ref, self.target_ref, self.goal) == (skill.skill_instance_id, skill.phase,
            skill.subject_ref, skill.source_ref, skill.target_ref, goal)


@dataclass(frozen=True, slots=True)
class LocalIntentConditionV1:
    """局部完成定义；字符串 required 引用同次 Runtime 反馈的具名测量。"""
    measurement: str
    operator: str
    required: float | bool | str

    def __post_init__(self):
        if not self.measurement.strip() or self.operator not in ("equals", "at_least", "at_most"):
            raise ValueError("local intent condition requires a named measurement and comparison")
        if isinstance(self.required, str):
            if not self.required.strip():
                raise ValueError("local intent condition reference cannot be empty")
        elif not isfinite(self.required):
            raise ValueError("local intent condition target must be finite")


@dataclass(frozen=True, slots=True)
class LocalIntentIntervalV1:
    """具名米制测量的目标区间；不表示固定 TCP 点或释放许可。"""
    measurement: str
    minimum_m: float
    maximum_m: float

    def __post_init__(self):
        if (not self.measurement.strip() or not all(isfinite(v) for v in (self.minimum_m, self.maximum_m))
            or self.minimum_m > self.maximum_m):
            raise ValueError("local intent interval requires a named finite ordered metric range")


@dataclass(frozen=True, slots=True)
class PlacementMarginPreparationV1:
    """候选选择帧的余量与绑定；历史反事实不证明当前效果。"""
    task_id: str
    skill_instance_id: str
    subject_ref: str
    source_ref: str | None
    target_ref: str
    goal: tuple[str, str, bool]
    baseline_state_id: int
    baseline_action_epoch: int
    baseline_observation_id: int
    baseline_capture_timestamp_s: float
    scene_id: str
    sensor_id: str
    calibration_id: str
    region_boundary_xy_m: tuple[tuple[float, float], ...]
    baseline_tcp_robot_base_m: tuple[float, float, float]
    baseline_margin_m: float
    predicted_margin_m: float
    minimum_improvement_m: float

    def __post_init__(self):
        if any(not s for s in (self.task_id, self.skill_instance_id, self.subject_ref,
            self.target_ref, self.scene_id, self.sensor_id, self.calibration_id)):
            raise ValueError("placement preparation requires identified evidence")
        if (self.goal not in (("in_region", self.target_ref, True), ("inside", self.target_ref, True))
            or self.baseline_state_id <= 0 or self.baseline_action_epoch < 0 or self.baseline_observation_id <= 0
            or not isfinite(self.baseline_capture_timestamp_s) or self.baseline_capture_timestamp_s < 0
            or len(self.region_boundary_xy_m) < 3
            or any(len(p)!=2 or not all(isfinite(x) for x in p) for p in self.region_boundary_xy_m)
            or len(self.baseline_tcp_robot_base_m)!=3 or not all(isfinite(x) for x in self.baseline_tcp_robot_base_m)
            or not all(isfinite(x) for x in (self.baseline_margin_m, self.predicted_margin_m, self.minimum_improvement_m))
            or self.baseline_margin_m < 0 or self.minimum_improvement_m <= 0
            or self.predicted_margin_m - self.baseline_margin_m + 1e-12 < self.minimum_improvement_m):
            raise ValueError("placement preparation requires a positive predicted native margin improvement")


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
    completion_conditions: tuple[LocalIntentConditionV1, ...] = ()
    progress_interval: LocalIntentIntervalV1 | None = None
    placement_preparation: PlacementMarginPreparationV1 | None = None

    def __post_init__(self):
        if not self.choice_id.strip() or not self.objective.strip() or not self.allowed_actions:
            raise ValueError("local intent requires an identity, objective and allowed actions")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("local intent actions must be unique")
        if self.metric_kind not in ("tcp_xy", "tcp_z", "phase_contract", "contact_geometry", "runtime_conditions"):
            raise ValueError("unknown local intent metric")
        if (self.metric_kind == "runtime_conditions") != bool(self.completion_conditions):
            raise ValueError("Runtime-condition intent requires its own completion definition")
        if self.progress_interval is not None and self.metric_kind != "runtime_conditions":
            raise ValueError("interval progress requires Runtime-condition completion")
        if self.metric_kind == "contact_geometry" and self.contact_side is None:
            raise ValueError("contact intent requires a bound side")
        if self.metric_kind in ("tcp_xy", "tcp_z"):
            if (self.target_robot_base_m is None or len(self.target_robot_base_m) != 3
                or not all(isfinite(v) for v in self.target_robot_base_m)
                or self.tolerance_m is None or not isfinite(self.tolerance_m) or self.tolerance_m <= 0):
                raise ValueError("geometric intent requires a finite target and positive tolerance")
        if self.placement_preparation is not None and (
            self.metric_kind != "tcp_xy" or self.reference_ref != self.placement_preparation.target_ref
            or not self.choice_id.startswith("prepare_placement_margin_")
            or any(a not in ("forward", "backward", "left", "right", "hold") for a in self.allowed_actions)):
            raise ValueError("placement margin preparation requires a bound horizontal goal")


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
    origin: str = "jev_choice"
    plan_selection_state_id: int | None = None

    def __post_init__(self):
        if self.status not in ("active", "completed", "invalidated", "superseded", "cancelled"):
            raise ValueError("invalid local intent execution status")
        if (self.intent_id <= 0 or self.selected_state_id <= 0 or self.selected_action_epoch < 0
            or self.nonprogress_actions < 0 or self.last_observed_action_epoch < self.selected_action_epoch):
            raise ValueError("invalid local intent execution counters")
        for value in (self.metric_m, self.best_metric_m):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("local intent metric must be nonnegative or unknown")
        if self.origin not in ("jev_choice", "push_plan_step") or (
            self.origin == "push_plan_step" and self.plan_selection_state_id is None):
            raise ValueError("derived step requires its original model plan selection")


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
    push_contact_plan: PushContactPlanV1 | None = None
    push_preparation: PushPreparationV1 | None = None
    push_goal_execution: PushGoalExecutionV1 | None = None

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
        if self.push_contact_plan is not None and (
            self.push_contact_plan.skill_instance_id != self.skill_instance_id
            or self.phase != "push_toward_goal"
            or self.push_contact_plan.selected_state_id > self.based_on_state_id
            or self.push_contact_plan.selected_action_epoch > self.based_on_action_epoch
            or self.push_contact_plan.selected_observation_id > self.based_on_observation_id):
            raise ValueError("push plan must precede the current bound pushing context")
        if self.push_preparation is not None:
            plan = self.push_preparation.plan
            if (plan.skill_instance_id != self.skill_instance_id or self.phase != "push_toward_goal"
                or plan.selected_state_id > self.based_on_state_id
                or plan.selected_action_epoch > self.based_on_action_epoch
                or plan.selected_observation_id > self.based_on_observation_id):
                raise ValueError("push preparation must precede its bound current context")
        if self.push_goal_execution is not None and (
            self.push_goal_execution.plan.skill_instance_id != self.skill_instance_id
            or self.phase != "push_toward_goal"
            or self.push_goal_execution.plan.selected_state_id > self.based_on_state_id
            or self.push_goal_execution.last_counted_action_epoch > self.based_on_action_epoch
            or self.push_goal_execution.plan.selected_observation_id > self.based_on_observation_id):
            raise ValueError("persistent goal plan must belong to the current pushing scope")
        if self.push_goal_execution is not None:
            for observation in (self.push_goal_execution.baseline, self.push_goal_execution.current_goal,
                self.push_goal_execution.best_goal):
                if observation is not None and (observation.state_id > self.based_on_state_id
                    or observation.action_epoch > self.based_on_action_epoch
                    or observation.observation_id > self.based_on_observation_id):
                    raise ValueError("push progress cannot use future observations")
            if any(p.state_id > self.based_on_state_id or p.action_epoch > self.based_on_action_epoch
                or p.observation_id > self.based_on_observation_id
                for p in self.push_goal_execution.best_exit_progress):
                raise ValueError("exit progress history cannot originate from a future snapshot")
