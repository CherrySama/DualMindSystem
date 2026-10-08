from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Any
from jev4mujoco.contracts.local_intent import LocalIntentContextV1
from jev4mujoco.contracts.tool_geometry import ToolGeometryV1

from jev4mujoco.contracts.task_plan import TASK_REGISTRY_VERSION, TaskKind
from jev4mujoco.contracts.visual_scene import (
    QuatWxyz,
    Vec3,
    VisualPredicate,
    VisualSceneV1,
)


Joint6 = tuple[float, float, float, float, float, float]


class TaskRunStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    AWAITING_VERIFICATION = "awaiting_verification"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class SkillRunStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    AWAITING_VERIFICATION = "awaiting_verification"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class RobotExecutionState(str, Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    SETTLING = "settling"
    FAULT = "fault"


class GripperMotionState(str, Enum):
    OPENING = "opening"
    CLOSING = "closing"
    STOPPED = "stopped"


class GraspState(str, Enum):
    UNKNOWN = "unknown"
    NOT_HELD = "not_held"
    CANDIDATE_HELD = "candidate_held"
    HELD = "held"


class EffectorContactMode(str, Enum):
    UNKNOWN = "unknown"
    CLEAR = "clear"
    SINGLE_FINGER = "single_finger"
    GRASP_CONTACT = "grasp_contact"
    PUSH_CONTACT = "push_contact"


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _finite_vector(value: tuple[float, ...], size: int, name: str) -> None:
    if len(value) != size or not all(isfinite(component) for component in value):
        raise ValueError(f"{name} must contain {size} finite values")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _to_jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class StateSourcesV2:
    visual_scene_id: str
    visual_observation_id: int
    visual_capture_timestamp_s: float
    robot_measurement_timestamp_s: float
    task_plan_id: str
    task_registry_version: str

    def __post_init__(self) -> None:
        for name, value in (
            ("visual_scene_id", self.visual_scene_id),
            ("task_plan_id", self.task_plan_id),
            ("task_registry_version", self.task_registry_version),
        ):
            _identifier(value, f"sources.{name}")
        if type(self.visual_observation_id) is not int or self.visual_observation_id <= 0:
            raise ValueError("visual_observation_id must be a positive integer")
        if (
            not isfinite(self.visual_capture_timestamp_s)
            or self.visual_capture_timestamp_s < 0.0
        ):
            raise ValueError("visual_capture_timestamp_s must be nonnegative")
        if (
            not isfinite(self.robot_measurement_timestamp_s)
            or self.robot_measurement_timestamp_s < 0.0
        ):
            raise ValueError("robot_measurement_timestamp_s must be nonnegative")


@dataclass(frozen=True, slots=True)
class TaskStateV2:
    task_id: str
    task_kind: TaskKind
    status: TaskRunStatus
    completed_subject_refs: tuple[str, ...]
    pending_subject_refs: tuple[str, ...]
    current_subject_ref: str | None
    current_goal_ref: str | None

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task.task_id")
        if set(self.completed_subject_refs) & set(self.pending_subject_refs):
            raise ValueError("completed and pending task subjects must be disjoint")
        for reference in (
            *self.completed_subject_refs,
            *self.pending_subject_refs,
        ):
            _identifier(reference, "task subject reference")
        if self.current_subject_ref is not None:
            _identifier(self.current_subject_ref, "task.current_subject_ref")
        if self.current_goal_ref is not None:
            _identifier(self.current_goal_ref, "task.current_goal_ref")


@dataclass(frozen=True, slots=True)
class SkillGoalV1:
    predicate: VisualPredicate
    reference_ref: str
    desired_value: bool

    def __post_init__(self) -> None:
        _identifier(self.reference_ref, "skill.goal.reference_ref")
        if type(self.desired_value) is not bool:
            raise ValueError("skill.goal.desired_value must be boolean")


@dataclass(frozen=True, slots=True)
class SkillStateV2:
    skill_instance_id: str
    skill_kind: str
    phase: str
    subject_ref: str | None
    target_ref: str | None
    allowed_intents: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    status: SkillRunStatus
    goal: SkillGoalV1 | None = None
    source_ref: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("skill_instance_id", self.skill_instance_id),
            ("skill_kind", self.skill_kind),
            ("phase", self.phase),
        ):
            _identifier(value, f"skill.{name}")
        if self.subject_ref is not None:
            _identifier(self.subject_ref, "skill.subject_ref")
        if self.target_ref is not None:
            _identifier(self.target_ref, "skill.target_ref")
        if self.source_ref is not None:
            _identifier(self.source_ref, "skill.source_ref")
        if len(self.allowed_intents) != len(set(self.allowed_intents)):
            raise ValueError("allowed_intents must be unique")
        if len(self.allowed_actions) != len(set(self.allowed_actions)):
            raise ValueError("allowed_actions must be unique")
        for name, values in (
            ("allowed_intents", self.allowed_intents),
            ("allowed_actions", self.allowed_actions),
        ):
            for value in values:
                _identifier(value, f"skill.{name} item")
        if self.status is SkillRunStatus.RUNNING and not self.allowed_intents:
            raise ValueError("running skill must expose at least one allowed intent")


@dataclass(frozen=True, slots=True)
class GripperStateV2:
    opening_m: float
    target_opening_m: float
    motion_state: GripperMotionState
    right_joint_position_m: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.opening_m) or self.opening_m < 0.0:
            raise ValueError("gripper.opening_m must be finite and nonnegative")
        if not isfinite(self.target_opening_m) or self.target_opening_m < 0.0:
            raise ValueError("gripper.target_opening_m must be finite and nonnegative")
        if self.right_joint_position_m is not None and not isfinite(self.right_joint_position_m):
            raise ValueError("right finger position must be finite or unknown")


@dataclass(frozen=True, slots=True)
class RobotStateV2:
    joint_positions_rad: Joint6
    joint_velocities_rad_s: Joint6
    tcp_position_robot_base_m: Vec3
    tcp_orientation_robot_base_wxyz: QuatWxyz
    gripper: GripperStateV2
    execution_state: RobotExecutionState
    tool_geometry: ToolGeometryV1 | None = None

    def __post_init__(self) -> None:
        _finite_vector(self.joint_positions_rad, 6, "robot.joint_positions_rad")
        _finite_vector(self.joint_velocities_rad_s, 6, "robot.joint_velocities_rad_s")
        _finite_vector(self.tcp_position_robot_base_m, 3, "robot.tcp_position_robot_base_m")
        _finite_vector(
            self.tcp_orientation_robot_base_wxyz,
            4,
            "robot.tcp_orientation_robot_base_wxyz",
        )


@dataclass(frozen=True, slots=True)
class SupportStateV2:
    subject_ref: str | None
    support_region_ref: str | None
    supported: bool | None

    def __post_init__(self) -> None:
        if self.supported is not None and type(self.supported) is not bool:
            raise ValueError("supported must be boolean or unknown")
        if self.subject_ref is not None:
            _identifier(self.subject_ref, "support.subject_ref")
        if self.support_region_ref is not None:
            _identifier(self.support_region_ref, "support.support_region_ref")
        if self.supported and (
            self.subject_ref is None or self.support_region_ref is None
        ):
            raise ValueError("supported state requires subject and support region")


@dataclass(frozen=True, slots=True)
class PhysicalEvidenceV2:
    robot_feedback_available: bool
    visual_follow_confirmed: bool
    fresh_visual_required: bool
    grasp_probe_lifted: bool = False
    grasp_probe_failed: bool = False
    contact_source: str = "unknown"
    grasp_source: str = "unknown"
    support_source: str = "unknown"

    def __post_init__(self) -> None:
        for name in ("contact_source", "grasp_source", "support_source"):
            _identifier(getattr(self, name), f"physical evidence {name}")


@dataclass(frozen=True, slots=True)
class EffectorObjectContactV2:
    subject_ref: str | None
    mode: EffectorContactMode
    left_finger_contact: bool | None
    right_finger_contact: bool | None
    support_contact: bool | None

    def __post_init__(self) -> None:
        for name in ("left_finger_contact", "right_finger_contact", "support_contact"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be boolean or unknown")
        if self.subject_ref is not None:
            _identifier(self.subject_ref, "physical.effector_contact.subject_ref")
        if self.mode is EffectorContactMode.CLEAR and (
            self.left_finger_contact or self.right_finger_contact
        ):
            raise ValueError("clear contact mode cannot report finger contact")
        if self.mode in (
            EffectorContactMode.GRASP_CONTACT,
            EffectorContactMode.PUSH_CONTACT,
        ) and self.subject_ref is None:
            raise ValueError("active contact mode requires a subject_ref")
        if self.mode is EffectorContactMode.PUSH_CONTACT and not self.support_contact:
            raise ValueError("push contact requires maintained support contact")


@dataclass(frozen=True, slots=True)
class PhysicalStateV2:
    grasp_state: GraspState
    held_object_ref: str | None
    effector_contact: EffectorObjectContactV2
    support_state: SupportStateV2
    evidence: PhysicalEvidenceV2

    def __post_init__(self) -> None:
        if self.held_object_ref is not None:
            _identifier(self.held_object_ref, "physical.held_object_ref")
        if self.grasp_state is GraspState.HELD and self.held_object_ref is None:
            raise ValueError("held grasp state requires held_object_ref")
        if self.grasp_state in (GraspState.NOT_HELD, GraspState.UNKNOWN) and self.held_object_ref is not None:
            raise ValueError("not-held or unknown grasp state cannot name a held object")


@dataclass(frozen=True, slots=True)
class LocalGoalV2:
    goal_type: str
    target_relation: str

    def __post_init__(self) -> None:
        _identifier(self.goal_type, "interaction.local_goal.goal_type")
        _identifier(self.target_relation, "interaction.local_goal.target_relation")


@dataclass(frozen=True, slots=True)
class DerivedInteractionV2:
    subject_from_tcp_robot_base_m: Vec3
    distance_m: float
    dominant_axis: str | None

    def __post_init__(self) -> None:
        _finite_vector(
            self.subject_from_tcp_robot_base_m,
            3,
            "interaction.subject_from_tcp_robot_base_m",
        )
        if not isfinite(self.distance_m) or self.distance_m < 0.0:
            raise ValueError("interaction.distance_m must be finite and nonnegative")
        if self.dominant_axis not in (None, "x", "y", "z"):
            raise ValueError("dominant_axis must be x, y, z or None")


@dataclass(frozen=True, slots=True)
class InteractionStateV2:
    subject_ref: str | None
    reference_ref: str | None
    local_goal: LocalGoalV2 | None
    derived: DerivedInteractionV2 | None

    def __post_init__(self) -> None:
        if self.subject_ref is not None:
            _identifier(self.subject_ref, "interaction.subject_ref")
        if self.reference_ref is not None:
            _identifier(self.reference_ref, "interaction.reference_ref")
        if (self.local_goal is None) != (self.derived is None):
            raise ValueError("local_goal and derived interaction must appear together")


@dataclass(frozen=True, slots=True)
class FreshnessStateV2:
    visual_age_s: float
    visual_is_fresh: bool
    prediction_steps: int
    pending_contact_verification: bool
    pending_completion_verification: bool

    def __post_init__(self) -> None:
        if not isfinite(self.visual_age_s) or self.visual_age_s < 0.0:
            raise ValueError("visual_age_s must be finite and nonnegative")
        if type(self.prediction_steps) is not int or self.prediction_steps < 0:
            raise ValueError("prediction_steps must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class ActionFeedbackV1:
    decision_id: int
    action: str
    start_action_epoch: int
    end_action_epoch: int
    status: str
    commanded_delta_robot_base_m: Vec3
    measured_delta_robot_base_m: Vec3
    end_timestamp_s: float

    def __post_init__(self) -> None:
        if type(self.decision_id) is not int or self.decision_id <= 0:
            raise ValueError("decision_id must be a positive integer")
        _identifier(self.action, "recent_action.action")
        _identifier(self.status, "recent_action.status")
        if (
            type(self.start_action_epoch) is not int
            or type(self.end_action_epoch) is not int
            or self.start_action_epoch < 0
            or self.end_action_epoch <= self.start_action_epoch
        ):
            raise ValueError("action feedback must advance action_epoch")
        _finite_vector(
            self.commanded_delta_robot_base_m,
            3,
            "recent_action.commanded_delta_robot_base_m",
        )
        _finite_vector(
            self.measured_delta_robot_base_m,
            3,
            "recent_action.measured_delta_robot_base_m",
        )
        if not isfinite(self.end_timestamp_s) or self.end_timestamp_s < 0.0:
            raise ValueError("recent_action.end_timestamp_s must be nonnegative")


@dataclass(frozen=True, slots=True)
class DecisionFeedbackV1:
    decision_id: int
    action: str
    phase: str
    status: str
    reason: str
    error_before_m: float | None
    projected_error_after_m: float | None
    error_after_m: float | None = None
    progress_m: float | None = None

    def __post_init__(self) -> None:
        if type(self.decision_id) is not int or self.decision_id <= 0:
            raise ValueError("decision feedback requires a positive decision_id")
        for name in ("action", "phase", "reason"):
            _identifier(getattr(self, name), f"decision_feedback.{name}")
        if self.status not in ("accepted", "rejected", "observed"):
            raise ValueError("decision feedback has an unknown status")
        for name in ("error_before_m", "projected_error_after_m"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError(f"decision_feedback.{name} must be nonnegative")
        if self.status == "observed":
            if (
                self.error_before_m is None
                or self.projected_error_after_m is None
                or self.error_after_m is None
                or self.progress_m is None
            ):
                raise ValueError("observed decision feedback requires measured progress")
            if not isfinite(self.error_after_m) or self.error_after_m < 0.0:
                raise ValueError("decision_feedback.error_after_m must be nonnegative")
            if not isfinite(self.progress_m):
                raise ValueError("decision_feedback.progress_m must be finite")
        elif self.error_after_m is not None or self.progress_m is not None:
            raise ValueError("unexecuted decision feedback cannot report measured progress")


@dataclass(frozen=True, slots=True)
class PhaseMeasurementV1:
    """具名观测量；没有数据时必须是 None，而不是零。"""

    name: str
    value: float | bool | tuple[float, ...] | None
    unit: str
    source: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("name", "unit", "source"):
            _identifier(getattr(self, name), f"phase measurement {name}")
        values = self.value if isinstance(self.value, tuple) else (self.value,)
        if any(value is not None and not isfinite(value) for value in values):
            raise ValueError("phase measurement values must be finite or unknown")
        if self.confidence is not None and (not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0):
            raise ValueError("phase measurement confidence must be in [0, 1] or uncalibrated")


@dataclass(frozen=True, slots=True)
class PhaseConditionV1:
    measurement: str
    observed: float | bool | None
    operator: str
    required: float | bool | None
    tolerance: float | None = None
    evidence_current: bool = True

    def __post_init__(self) -> None:
        _identifier(self.measurement, "phase condition measurement")
        if self.operator not in ("equals", "within", "at_most", "at_least", "greater_than"):
            raise ValueError("unknown phase condition operator")
        if any(value is not None and not isfinite(value) for value in (self.observed, self.required, self.tolerance)):
            raise ValueError("phase condition numbers must be finite or unknown")
        if self.tolerance is not None and self.tolerance < 0.0:
            raise ValueError("phase condition tolerance cannot be negative")
        if type(self.evidence_current) is not bool:
            raise ValueError("phase condition evidence_current must be boolean")

    @property
    def satisfied(self) -> bool | None:
        if not self.evidence_current or self.observed is None or self.required is None:
            return None
        if self.operator == "within":
            return None if self.tolerance is None else abs(self.observed - self.required) <= self.tolerance
        if self.operator == "at_most":
            return self.observed <= self.required
        if self.operator == "at_least":
            return self.observed >= self.required
        if self.operator == "greater_than":
            return self.observed > self.required
        return self.observed == self.required


@dataclass(frozen=True, slots=True)
class PhaseEvidenceBasisV1:
    """反馈所依观测的时间，不能用反馈生成时间刷新旧证据。"""

    visual_capture_timestamp_s: float
    robot_measurement_timestamp_s: float
    visual_evidence_current: bool
    robot_evidence_current: bool

    def __post_init__(self) -> None:
        for name in ("visual_capture_timestamp_s", "robot_measurement_timestamp_s"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError("phase evidence timestamps must be finite and nonnegative")
        for name in ("visual_evidence_current", "robot_evidence_current"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("phase evidence freshness must be boolean")


@dataclass(frozen=True, slots=True)
class PhaseObservationBasisV1:
    """测量生成时的身份与有效性；历史不能借当前观测刷新依据。"""

    state_id: int
    action_epoch: int
    observation_id: int
    timestamp_s: float
    scene_id: str
    sensor_id: str
    calibration_id: str
    skill: SkillStateV2
    evidence_basis: PhaseEvidenceBasisV1
    measurement_evidence_current: tuple[tuple[str, bool], ...]
    preceding_action_end_timestamp_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("state_id", "observation_id"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError("phase observation IDs must be positive integers")
        if type(self.action_epoch) is not int or self.action_epoch < 0:
            raise ValueError("phase observation action epoch must be nonnegative")
        for name in ("scene_id", "sensor_id", "calibration_id"):
            _identifier(getattr(self, name), f"phase observation {name}")
        if self.timestamp_s is None:
            raise ValueError("phase observation snapshot time is required")
        for value in (self.timestamp_s, self.preceding_action_end_timestamp_s):
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError("phase observation times must be finite and nonnegative")
        names = [name for name, _ in self.measurement_evidence_current]
        if len(set(names)) != len(names):
            raise ValueError("phase observation measurement names must be unique")
        for name, valid in self.measurement_evidence_current:
            _identifier(name, "phase observation measurement")
            if type(valid) is not bool:
                raise ValueError("phase observation measurement validity must be boolean")

    def same_scope_as(self, other: SkillStateV2) -> bool:
        return self.same_scope(self.skill, other)

    @staticmethod
    def same_scope(first: SkillStateV2 | None, second: SkillStateV2) -> bool:
        # 同一阶段的合法动作许可可变化，不改变主体、目标与任务语义绑定。
        names = ("skill_instance_id", "skill_kind", "phase", "subject_ref", "source_ref", "target_ref", "goal")
        return first is not None and all(getattr(first, name) == getattr(second, name) for name in names)


@dataclass(frozen=True, slots=True)
class PhaseConstraintFeedbackV1:
    """当前阶段的约束诊断；不产生动作许可或阶段转换。"""

    constraint_id: str
    scope: str
    status: str
    reason: str
    condition: PhaseConditionV1 | None = None
    evidence: tuple[PhaseMeasurementV1, ...] = ()

    def __post_init__(self) -> None:
        for name in ("constraint_id", "reason"):
            _identifier(getattr(self, name), f"phase constraint {name}")
        if self.scope not in ("active_phase", "until_observed_release"):
            raise ValueError("unknown phase constraint scope")
        if self.status not in ("satisfied", "violated", "unknown", "unsupported", "not_applicable"):
            raise ValueError("unknown phase constraint status")
        if self.status in ("satisfied", "violated", "unknown"):
            if self.condition is None:
                raise ValueError("supported constraint requires its evaluated condition")
            expected = {True: "satisfied", False: "violated", None: "unknown"}[self.condition.satisfied]
            if self.status != expected:
                raise ValueError("constraint status does not match its condition")
        elif self.condition is not None or self.evidence:
            raise ValueError("unsupported or inactive constraint cannot certify evidence")


@dataclass(frozen=True, slots=True)
class PhaseFeedbackV1:
    skill_instance_id: str
    phase: str
    contract_id: str
    based_on_state_id: int
    based_on_action_epoch: int
    based_on_observation_id: int
    outcome: str
    reason: str
    measurements: tuple[PhaseMeasurementV1, ...] = ()
    parameters: tuple[tuple[str, float | int], ...] = ()
    previous_measurements: tuple[PhaseMeasurementV1, ...] = ()
    previous_state_id: int | None = None
    previous_action_epoch: int | None = None
    consecutive_pass_count: int = 0
    required_consecutive_passes: int = 1
    conditions: tuple[PhaseConditionV1, ...] = ()
    maintain_constraints: tuple[PhaseConstraintFeedbackV1, ...] = ()
    evidence_basis: PhaseEvidenceBasisV1 | None = None
    based_on_skill: SkillStateV2 | None = None
    observation_basis: PhaseObservationBasisV1 | None = None
    previous_observation_basis: PhaseObservationBasisV1 | None = None

    def __post_init__(self) -> None:
        for name in ("skill_instance_id", "phase", "contract_id", "reason"):
            _identifier(getattr(self, name), f"phase feedback {name}")
        if self.outcome not in ("pending", "passed", "failed"):
            raise ValueError("unknown phase feedback outcome")
        if len({item.name for item in self.measurements}) != len(self.measurements):
            raise ValueError("phase measurement names must be unique")
        if len({item.constraint_id for item in self.maintain_constraints}) != len(self.maintain_constraints):
            raise ValueError("phase constraint IDs must be unique")
        if self.maintain_constraints and self.evidence_basis is None:
            raise ValueError("phase constraints require their observation evidence basis")
        if any(not isfinite(value) for _, value in self.parameters):
            raise ValueError("phase parameters must be finite")
        if self.previous_measurements and (
            self.previous_state_id is None or self.previous_action_epoch is None
            or self.previous_action_epoch >= self.based_on_action_epoch
        ):
            raise ValueError("previous phase measurements must precede the current action epoch")
        if self.observation_basis is not None:
            basis = self.observation_basis
            if basis.evidence_basis != self.evidence_basis:
                raise ValueError("phase observation evidence timestamps or validity do not match its feedback")
            if ((basis.state_id, basis.action_epoch, basis.observation_id)
                != (self.based_on_state_id, self.based_on_action_epoch, self.based_on_observation_id)
                or basis.skill != self.based_on_skill
                or set(dict(basis.measurement_evidence_current)) != {item.name for item in self.measurements}):
                raise ValueError("phase observation basis does not match its feedback")
        if self.previous_observation_basis is not None:
            basis = self.previous_observation_basis
            if (not self.previous_measurements or self.based_on_skill is None
                or (basis.state_id, basis.action_epoch) != (self.previous_state_id, self.previous_action_epoch)
                or not basis.same_scope_as(self.based_on_skill)
                or set(dict(basis.measurement_evidence_current)) != {item.name for item in self.previous_measurements}):
                raise ValueError("previous observation basis does not match its phase measurements")


@dataclass(frozen=True, slots=True)
class StateSnapshotV2:
    state_id: int
    action_epoch: int
    timestamp_s: float
    sources: StateSourcesV2
    task: TaskStateV2
    skill: SkillStateV2
    robot: RobotStateV2
    physical: PhysicalStateV2
    visual_scene: VisualSceneV1
    interaction: InteractionStateV2
    freshness: FreshnessStateV2
    recent_action: ActionFeedbackV1 | None
    decision_feedback: DecisionFeedbackV1 | None = None
    phase_feedback: PhaseFeedbackV1 | None = None
    local_intent: LocalIntentContextV1 | None = None
    schema: str = "StateSnapshotV2"
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema != "StateSnapshotV2" or self.schema_version != 2:
            raise ValueError("StateSnapshotV2 schema identity is fixed")
        if type(self.state_id) is not int or self.state_id <= 0:
            raise ValueError("state_id must be a positive integer")
        if type(self.action_epoch) is not int or self.action_epoch < 0:
            raise ValueError("action_epoch must be a nonnegative integer")
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and nonnegative")
        if self.sources.visual_scene_id != self.visual_scene.scene_id:
            raise ValueError("sources.visual_scene_id does not match visual_scene")
        if self.sources.visual_observation_id != self.visual_scene.observation_id:
            raise ValueError("sources.visual_observation_id does not match visual_scene")
        if (
            self.sources.visual_capture_timestamp_s
            != self.visual_scene.capture_timestamp_s
        ):
            raise ValueError("visual capture timestamp does not match visual_scene")
        if self.task.task_id != self.sources.task_plan_id:
            raise ValueError("task state does not match sources.task_plan_id")
        if self.sources.task_registry_version != TASK_REGISTRY_VERSION:
            raise ValueError("unsupported task registry version")
        if self.sources.robot_measurement_timestamp_s > self.timestamp_s:
            raise ValueError("robot measurement cannot be newer than the snapshot")
        if self.visual_scene.capture_timestamp_s > self.timestamp_s:
            raise ValueError("visual capture cannot be newer than the snapshot")
        expected_visual_age = self.timestamp_s - self.visual_scene.capture_timestamp_s
        if abs(self.freshness.visual_age_s - expected_visual_age) > 1.0e-9:
            raise ValueError("freshness.visual_age_s does not match snapshot timestamps")
        if self.recent_action is not None and self.recent_action.end_action_epoch != self.action_epoch:
            raise ValueError("recent action end epoch must equal snapshot action_epoch")
        if self.phase_feedback is not None:
            feedback = self.phase_feedback
            if (feedback.based_on_state_id != self.state_id
                or feedback.based_on_action_epoch != self.action_epoch
                or feedback.based_on_observation_id != self.visual_scene.observation_id
                or feedback.skill_instance_id != self.skill.skill_instance_id
                or feedback.phase != self.skill.phase):
                raise ValueError("phase feedback does not match the current snapshot")
            if feedback.based_on_skill is not None and feedback.based_on_skill != self.skill:
                raise ValueError("phase feedback scope does not match the current skill")
            basis = feedback.evidence_basis
            if basis is not None and (
                basis.visual_capture_timestamp_s != self.sources.visual_capture_timestamp_s
                or basis.robot_measurement_timestamp_s != self.sources.robot_measurement_timestamp_s
            ):
                raise ValueError("phase evidence timestamps do not match the current snapshot")
            observation = feedback.observation_basis
            if observation is not None and (
                observation.timestamp_s != self.timestamp_s
                or observation.scene_id != self.visual_scene.scene_id
                or observation.sensor_id != self.visual_scene.sensor.sensor_id
                or observation.calibration_id != self.visual_scene.sensor.calibration_id
                or observation.preceding_action_end_timestamp_s != (
                    None if self.recent_action is None else self.recent_action.end_timestamp_s)
            ):
                raise ValueError("phase observation basis does not match the current snapshot")
        if self.local_intent is not None:
            intent = self.local_intent
            if (intent.based_on_state_id != self.state_id
                or intent.based_on_action_epoch != self.action_epoch
                or intent.based_on_observation_id != self.visual_scene.observation_id
                or intent.skill_instance_id != self.skill.skill_instance_id
                or intent.phase != self.skill.phase):
                raise ValueError("local intent context does not match the current snapshot")

        entity_ids = {entity.track_id for entity in self.visual_scene.entities}
        region_ids = {region.region_id for region in self.visual_scene.regions}
        all_ids = entity_ids | region_ids
        for name, reference in (
            ("task.current_subject_ref", self.task.current_subject_ref),
            ("task.current_goal_ref", self.task.current_goal_ref),
            ("skill.subject_ref", self.skill.subject_ref),
            ("skill.target_ref", self.skill.target_ref),
            ("skill.source_ref", self.skill.source_ref),
            ("physical.held_object_ref", self.physical.held_object_ref),
            (
                "physical.effector_contact.subject_ref",
                self.physical.effector_contact.subject_ref,
            ),
            ("interaction.subject_ref", self.interaction.subject_ref),
            ("interaction.reference_ref", self.interaction.reference_ref),
            (
                "skill.goal.reference_ref",
                None if self.skill.goal is None else self.skill.goal.reference_ref,
            ),
        ):
            if reference is not None and reference not in all_ids:
                raise ValueError(f"{name} references an item absent from visual_scene")
        for reference in (
            *self.task.completed_subject_refs,
            *self.task.pending_subject_refs,
        ):
            if reference not in entity_ids:
                raise ValueError("task progress references an entity absent from visual_scene")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)
