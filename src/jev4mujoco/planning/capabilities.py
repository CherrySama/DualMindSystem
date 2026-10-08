"""通用能力类型、模板和程序契约；模板与实例校验保持在同一模块。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from jev4mujoco.contracts.actions import AtomicAction
from jev4mujoco.contracts.visual_scene import VisualPredicate

class CapabilityKind(str, Enum):
    PICK = "pick"
    TRANSPORT = "transport"
    PLACE = "place"
    SURFACE_PUSH = "surface_push"
    PRESS = "press"


class CapabilityRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_VERIFICATION = "awaiting_verification"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class CapabilityProgramStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class VerificationOutcome(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CapabilityPhaseTemplateV1:
    phase_id: str
    allowed_intents: tuple[str, ...]
    allowed_actions: tuple[AtomicAction, ...]
    completion_contract: str
    requires_fresh_visual: bool
    motor_action_allowed: bool = True

    def __post_init__(self) -> None:
        _identifier(self.phase_id, "phase_id")
        _identifier(self.completion_contract, "phase completion_contract")
        if not self.allowed_intents:
            raise ValueError("capability phase must expose at least one intent")
        if len(self.allowed_intents) != len(set(self.allowed_intents)):
            raise ValueError("phase allowed_intents must be unique")
        if len(self.allowed_actions) != len(set(self.allowed_actions)):
            raise ValueError("phase allowed_actions must be unique")
        if self.motor_action_allowed and not self.allowed_actions:
            raise ValueError("motor phase must expose at least one atomic action")
        if not self.motor_action_allowed and self.allowed_actions:
            raise ValueError("verification-only phase cannot expose motor actions")


@dataclass(frozen=True, slots=True)
class CapabilityTemplateV1:
    template_id: str
    kind: CapabilityKind
    entry_contract: str
    phases: tuple[CapabilityPhaseTemplateV1, ...]
    completion_contract: str
    runtime_implemented: bool = True

    def __post_init__(self) -> None:
        _identifier(self.template_id, "template_id")
        _identifier(self.entry_contract, "entry_contract")
        _identifier(self.completion_contract, "completion_contract")
        if not self.phases:
            raise ValueError("capability template requires at least one phase")
        phase_ids = [phase.phase_id for phase in self.phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("capability phase IDs must be unique within a template")
        if self.phases[-1].completion_contract != self.completion_contract:
            raise ValueError(
                "the final phase must use the capability completion contract"
            )


CAPABILITY_TEMPLATES_V1: dict[CapabilityKind, CapabilityTemplateV1] = {
    CapabilityKind.PICK: CapabilityTemplateV1(
        template_id="pick_v1",
        kind=CapabilityKind.PICK,
        entry_contract="subject_visible_and_supported_v1",
        phases=(
            CapabilityPhaseTemplateV1(
                phase_id="prepare_gripper",
                allowed_intents=("prepare_gripper",),
                allowed_actions=(AtomicAction.GRIPPER_OPEN, AtomicAction.HOLD),
                completion_contract="gripper_open_v1",
                requires_fresh_visual=False,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="align_subject_xy",
                allowed_intents=("align_subject_xy",),
                allowed_actions=(
                    AtomicAction.FORWARD,
                    AtomicAction.BACKWARD,
                    AtomicAction.LEFT,
                    AtomicAction.RIGHT,
                    AtomicAction.HOLD,
                ),
                completion_contract="tcp_subject_xy_aligned_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="descend_to_grasp",
                allowed_intents=("descend_to_grasp",),
                allowed_actions=(
                    AtomicAction.FORWARD,
                    AtomicAction.BACKWARD,
                    AtomicAction.LEFT,
                    AtomicAction.RIGHT,
                    AtomicAction.DOWN,
                    AtomicAction.UP,
                    AtomicAction.HOLD,
                ),
                completion_contract="tcp_grasp_pose_aligned_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="close_gripper",
                allowed_intents=("close_gripper",),
                allowed_actions=(AtomicAction.GRIPPER_CLOSE, AtomicAction.HOLD),
                completion_contract="grasp_candidate_v1",
                requires_fresh_visual=False,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="probe_grasp",
                allowed_intents=("probe_grasp",),
                allowed_actions=(AtomicAction.UP, AtomicAction.HOLD),
                completion_contract="grasp_probe_lifted_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="verify_grasp",
                allowed_intents=("request_grasp_verification",),
                allowed_actions=(),
                completion_contract="grasp_confirmed_v1",
                requires_fresh_visual=True,
                motor_action_allowed=False,
            ),
        ),
        completion_contract="grasp_confirmed_v1",
    ),
    CapabilityKind.TRANSPORT: CapabilityTemplateV1(
        template_id="transport_v1",
        kind=CapabilityKind.TRANSPORT,
        entry_contract="subject_held_v1",
        phases=(
            CapabilityPhaseTemplateV1(
                phase_id="lift_to_clearance",
                allowed_intents=("lift_to_clearance",),
                allowed_actions=(AtomicAction.UP,),
                completion_contract="transport_clearance_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="align_target_xy",
                allowed_intents=("align_target_xy",),
                allowed_actions=(
                    AtomicAction.FORWARD,
                    AtomicAction.BACKWARD,
                    AtomicAction.LEFT,
                    AtomicAction.RIGHT,
                    AtomicAction.HOLD,
                ),
                completion_contract="held_subject_above_region_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="verify_transport",
                allowed_intents=("request_transport_verification",),
                allowed_actions=(),
                completion_contract="transport_confirmed_v1",
                requires_fresh_visual=True,
                motor_action_allowed=False,
            ),
        ),
        completion_contract="transport_confirmed_v1",
    ),
    CapabilityKind.PLACE: CapabilityTemplateV1(
        template_id="place_v1",
        kind=CapabilityKind.PLACE,
        entry_contract="subject_held_above_region_v1",
        phases=(
            CapabilityPhaseTemplateV1(
                phase_id="lower_to_release",
                allowed_intents=("lower_to_release", "release_subject"),
                allowed_actions=(
                    AtomicAction.DOWN,
                    AtomicAction.UP,
                    AtomicAction.HOLD,
                    AtomicAction.GRIPPER_OPEN,
                ),
                completion_contract="release_candidate_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="verify_placement",
                allowed_intents=("request_placement_verification",),
                allowed_actions=(),
                completion_contract="place_goal_stable_v1",
                requires_fresh_visual=True,
                motor_action_allowed=False,
            ),
        ),
        completion_contract="place_goal_stable_v1",
    ),
    CapabilityKind.SURFACE_PUSH: CapabilityTemplateV1(
        template_id="surface_push_v1",
        kind=CapabilityKind.SURFACE_PUSH,
        entry_contract="subject_visible_supported_on_surface_v1",
        phases=(
            CapabilityPhaseTemplateV1(
                phase_id="prepare_effector",
                allowed_intents=("prepare_push_effector",),
                allowed_actions=(AtomicAction.GRIPPER_CLOSE, AtomicAction.HOLD),
                completion_contract="push_effector_ready_v1",
                requires_fresh_visual=False,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="align_precontact",
                allowed_intents=("align_push_precontact",),
                allowed_actions=(
                    AtomicAction.FORWARD,
                    AtomicAction.BACKWARD,
                    AtomicAction.LEFT,
                    AtomicAction.RIGHT,
                    AtomicAction.UP,
                    AtomicAction.DOWN,
                    AtomicAction.HOLD,
                ),
                completion_contract="push_precontact_aligned_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="establish_contact",
                allowed_intents=("establish_push_contact",),
                allowed_actions=(
                    AtomicAction.FORWARD,
                    AtomicAction.BACKWARD,
                    AtomicAction.LEFT,
                    AtomicAction.RIGHT,
                    AtomicAction.UP,
                    AtomicAction.DOWN,
                    AtomicAction.HOLD,
                ),
                completion_contract="push_contact_established_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="push_toward_goal",
                allowed_intents=("advance_surface_push",),
                allowed_actions=(
                    AtomicAction.FORWARD,
                    AtomicAction.BACKWARD,
                    AtomicAction.LEFT,
                    AtomicAction.RIGHT,
                    AtomicAction.UP,
                    AtomicAction.DOWN,
                    AtomicAction.HOLD,
                ),
                completion_contract="surface_push_goal_reached_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="retract_effector",
                allowed_intents=("retract_after_push",),
                allowed_actions=(
                    AtomicAction.FORWARD,
                    AtomicAction.BACKWARD,
                    AtomicAction.LEFT,
                    AtomicAction.RIGHT,
                    AtomicAction.UP,
                    AtomicAction.DOWN,
                    AtomicAction.HOLD,
                ),
                completion_contract="push_effector_retracted_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="verify_push_goal",
                allowed_intents=("request_surface_push_verification",),
                allowed_actions=(),
                completion_contract="surface_push_goal_stable_v1",
                requires_fresh_visual=True,
                motor_action_allowed=False,
            ),
        ),
        completion_contract="surface_push_goal_stable_v1",
        runtime_implemented=True,
    ),
    CapabilityKind.PRESS: CapabilityTemplateV1(
        template_id="press_v1",
        kind=CapabilityKind.PRESS,
        entry_contract="button_visible_v1",
        phases=(
            CapabilityPhaseTemplateV1(
                phase_id="prepare_effector",
                allowed_intents=("prepare_press_effector",),
                allowed_actions=(AtomicAction.GRIPPER_CLOSE, AtomicAction.HOLD),
                completion_contract="push_effector_ready_v1",
                requires_fresh_visual=False,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="align_press_point",
                allowed_intents=("align_press_point",),
                allowed_actions=(
                    AtomicAction.FORWARD,
                    AtomicAction.BACKWARD,
                    AtomicAction.LEFT,
                    AtomicAction.RIGHT,
                    AtomicAction.HOLD,
                ),
                completion_contract="press_point_aligned_v1",
                requires_fresh_visual=True,
            ),
            CapabilityPhaseTemplateV1(
                phase_id="press_button",
                allowed_intents=("press_along_surface_normal",),
                allowed_actions=(
                    AtomicAction.FORWARD,
                    AtomicAction.BACKWARD,
                    AtomicAction.LEFT,
                    AtomicAction.RIGHT,
                    AtomicAction.DOWN,
                    AtomicAction.HOLD,
                ),
                completion_contract="button_pressed_v1",
                requires_fresh_visual=True,
            ),
        ),
        completion_contract="button_pressed_v1",
    ),
}


@dataclass(frozen=True, slots=True)
class CapabilityGoalV1:
    predicate: VisualPredicate
    reference_ref: str
    desired_value: bool

    def __post_init__(self) -> None:
        _identifier(self.reference_ref, "goal.reference_ref")
        if type(self.desired_value) is not bool:
            raise ValueError("goal.desired_value must be boolean")


@dataclass(frozen=True, slots=True)
class CapabilityInstanceV1:
    instance_id: str
    template_id: str
    kind: CapabilityKind
    subject_ref: str
    source_ref: str | None
    target_ref: str | None
    ordinal: int
    goal: CapabilityGoalV1 | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("instance_id", self.instance_id),
            ("template_id", self.template_id),
            ("subject_ref", self.subject_ref),
        ):
            _identifier(value, name)
        if self.target_ref is not None:
            _identifier(self.target_ref, "target_ref")
        if self.source_ref is not None:
            _identifier(self.source_ref, "source_ref")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("capability ordinal must be a nonnegative integer")
        expected = CAPABILITY_TEMPLATES_V1[self.kind]
        if self.template_id != expected.template_id:
            raise ValueError("capability instance template does not match its kind")
        if self.kind in (
            CapabilityKind.PLACE,
            CapabilityKind.SURFACE_PUSH,
            CapabilityKind.PRESS,
        ):
            if self.goal is None:
                raise ValueError(f"{self.kind.value} capability requires a goal")
            if self.target_ref != self.goal.reference_ref:
                raise ValueError(
                    f"{self.kind.value} target_ref must match its goal reference"
                )
        elif self.kind is not CapabilityKind.TRANSPORT and self.goal is not None:
            raise ValueError("only goal-conditioned capabilities carry a V1 goal")


@dataclass(frozen=True, slots=True)
class CapabilityProgramV1:
    program_id: str
    task_id: str
    planning_scene_id: str
    steps: tuple[CapabilityInstanceV1, ...]
    schema: str = "CapabilityProgramV1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema != "CapabilityProgramV1" or self.schema_version != 1:
            raise ValueError("CapabilityProgramV1 schema identity is fixed")
        for name, value in (
            ("program_id", self.program_id),
            ("task_id", self.task_id),
            ("planning_scene_id", self.planning_scene_id),
        ):
            _identifier(value, name)
        if not self.steps:
            raise ValueError("capability program requires at least one step")
        instance_ids = [step.instance_id for step in self.steps]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("capability instance IDs must be unique")
        if tuple(step.ordinal for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("capability ordinals must be contiguous and ordered")


@dataclass(frozen=True, slots=True)
class CapabilityStepStateV1:
    instance_id: str
    status: CapabilityRunStatus
    phase_index: int

    def __post_init__(self) -> None:
        _identifier(self.instance_id, "instance_id")
        if type(self.phase_index) is not int or self.phase_index < 0:
            raise ValueError("phase_index must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class CapabilityExecutionStateV1:
    program_id: str
    status: CapabilityProgramStatus
    active_step_index: int | None
    steps: tuple[CapabilityStepStateV1, ...]
    transition_index: int
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.program_id, "program_id")
        if type(self.transition_index) is not int or self.transition_index < 0:
            raise ValueError("transition_index must be a nonnegative integer")
        if self.active_step_index is not None and (
            type(self.active_step_index) is not int
            or self.active_step_index < 0
            or self.active_step_index >= len(self.steps)
        ):
            raise ValueError("active_step_index is outside the step list")
        if self.status is CapabilityProgramStatus.BLOCKED:
            if self.blocked_reason is None:
                raise ValueError("blocked capability program requires a reason")
        elif self.blocked_reason is not None:
            raise ValueError("only blocked capability programs may carry a reason")
