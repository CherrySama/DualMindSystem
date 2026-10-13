from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, sqrt

from jev4mujoco.planning.capabilities import (
    CapabilityKind,
    CapabilityExecutionStateV1,
    CapabilityProgramStatus,
    CapabilityProgramV1,
    CapabilityRunStatus,
)
from jev4mujoco.runtime.transitions import active_skill_state_v2
from jev4mujoco.contracts.actions import ActionResult
from jev4mujoco.contracts.robot_feedback import RobotFeedbackV1
from jev4mujoco.runtime.interaction_estimator import (
    InteractionEstimatorConfigV1, VisualInteractionEstimatorV1, visual_support_estimate_v1,
)
from jev4mujoco.runtime.grasp_geometry import grasp_height_band_from_geometry_v1
from jev4mujoco.contracts.state_snapshot import (
    ActionFeedbackV1,
    DecisionFeedbackV1,
    DerivedInteractionV2,
    EffectorContactMode,
    EffectorObjectContactV2,
    FreshnessStateV2,
    GraspState,
    GraspDiagnosticV1,
    GripperMotionState,
    GripperStateV2,
    InteractionStateV2,
    LocalGoalV2,
    PhysicalEvidenceV2,
    PhysicalStateV2,
    RobotExecutionState,
    RobotStateV2,
    SkillRunStatus,
    SkillGoalV1,
    SkillStateV2,
    StateSnapshotV2,
    StateSourcesV2,
    SupportStateV2,
    TaskRunStatus,
    TaskStateV2,
)
from jev4mujoco.contracts.task_plan import TASK_REGISTRY_VERSION, TaskPlanV1
from jev4mujoco.contracts.visual_scene import (
    EntityV1,
    RegionV1,
    VisualSceneV1,
    Visibility,
)


Vec3 = tuple[float, float, float]


def _subtract(left: Vec3, right: Vec3) -> Vec3:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def _norm(vector: Vec3) -> float:
    return sqrt(sum(component * component for component in vector))


def _dominant_axis(vector: Vec3, deadband_m: float = 1.0e-6) -> str | None:
    magnitudes = tuple(abs(component) for component in vector)
    maximum = max(magnitudes)
    return None if maximum <= deadband_m else ("x", "y", "z")[magnitudes.index(maximum)]


@dataclass(frozen=True, slots=True)
class LiveStateConfigV2:
    gripper_open_joint_m: float
    gripper_closed_joint_m: float
    gripper_position_tolerance_m: float
    grasp_xy_tolerance_m: float = 0.008
    minimum_blocked_gripper_joint_m: float = 0.003
    grasp_probe_minimum_lift_m: float = 0.006
    grasp_probe_maximum_lift_m: float = 0.025
    grasp_probe_stable_hold_s: float = 0.15
    grasp_probe_maximum_relative_error_m: float = 0.006
    grasp_maximum_relative_drift_m: float = 0.018
    maximum_visual_age_s: float = 0.5
    support_height_tolerance_m: float = 0.01
    contact_geometry_gap_tolerance_m: float = 0.002
    minimum_interaction_motion_m: float = 0.0005


class LiveStateSnapshotBuilderV2:
    """融合公开 RGB-D 与机器人反馈；不接收仿真物体/接触快照。"""

    def __init__(self, config: LiveStateConfigV2, tool_calibration=None) -> None:
        self._config = config
        self._tool_calibration = tool_calibration
        self._interaction_estimator = VisualInteractionEstimatorV1(InteractionEstimatorConfigV1(
            config.contact_geometry_gap_tolerance_m, config.minimum_interaction_motion_m,
            config.grasp_probe_maximum_relative_error_m, config.maximum_visual_age_s))
        self._previous_subject_ref: str | None = None
        self._last_grasp_observation = None
        self._visual_follow_confirmed = False
        self._candidate_tcp: Vec3 | None = None
        self._candidate_centroid: Vec3 | None = None
        self._probe_lifted_at_s: float | None = None
        self._probe_relative: Vec3 | None = None
        self._grasp_confirmed = False
        self._grasp_probe_failed = False
        self._grasp_diagnostic_reason = "contact_candidate_absent"

    def build(
        self,
        *,
        robot_feedback: RobotFeedbackV1,
        visual_scene: VisualSceneV1,
        task_plan: TaskPlanV1,
        program: CapabilityProgramV1,
        execution: CapabilityExecutionStateV1,
        recent_action: ActionResult | None = None,
        decision_feedback: DecisionFeedbackV1 | None = None,
    ) -> StateSnapshotV2:
        if not isinstance(robot_feedback, RobotFeedbackV1):
            raise TypeError("state builder only accepts whitelisted robot feedback")
        raw = robot_feedback
        instance = self._current_instance(program, execution)
        if instance.subject_ref != self._previous_subject_ref:
            self._previous_subject_ref = instance.subject_ref
            self._reset_grasp_probe()
        subject = self._entity(visual_scene, instance.subject_ref)
        centroid = subject.geometry.centroid_robot_base_m.value
        extent = subject.geometry.extent_m.value
        tcp = raw.tcp_position_robot_base_m
        action_end = 0.0 if recent_action is None else recent_action.end_simulation_time_s
        current_visual = (0 <= raw.timestamp_s - visual_scene.capture_timestamp_s <= self._config.maximum_visual_age_s
            and visual_scene.capture_timestamp_s >= action_end)
        geometry_current = current_visual and subject.visibility is Visibility.CLEAR and centroid is not None and extent is not None
        tool = (self._tool_calibration.observe(tcp, raw.tcp_orientation_robot_base_wxyz,
            raw.gripper_joint_m, raw.right_gripper_joint_m) if self._tool_calibration is not None else None)
        interaction_estimate = self._interaction_estimator.observe(raw, visual_scene, subject, tool, recent_action)
        supported, visual_support_ref = visual_support_estimate_v1(visual_scene, subject,
            current=current_visual, height_tolerance_m=self._config.support_height_tolerance_m)
        gripper_commanded_closed = (
            abs(raw.gripper_target_m - self._config.gripper_closed_joint_m)
            <= self._config.gripper_position_tolerance_m
        )
        gripper_blocked = (
            raw.gripper_joint_m >= self._config.minimum_blocked_gripper_joint_m
        )
        push_mode = instance.kind in (CapabilityKind.SURFACE_PUSH, CapabilityKind.PRESS)
        band = None if not geometry_current else grasp_height_band_from_geometry_v1(centroid, extent)
        aligned = (band is not None and hypot(centroid[0] - tcp[0], centroid[1] - tcp[1]) <= self._config.grasp_xy_tolerance_m
            and band[0] <= tcp[2] <= band[1])
        grasp_contact_candidate = gripper_commanded_closed and gripper_blocked and geometry_current and (
            aligned or self._candidate_tcp is not None)
        # 遮挡只使当前证据未知，保留历史锚点以便下一帧重新核验；不沿用历史持有许可。
        observation_key = (visual_scene.scene_id, visual_scene.observation_id, instance.subject_ref)
        if not (gripper_commanded_closed and gripper_blocked):
            self._reset_grasp_probe()
        elif geometry_current and observation_key != self._last_grasp_observation:
            self._update_grasp_probe(centroid if geometry_current else None, tcp, raw.simulation_time_s,
                grasp_contact_candidate, None if extent is None else extent[2],
                instance.kind is CapabilityKind.PLACE and supported is True)
            self._last_grasp_observation = observation_key
        elif not geometry_current:
            self._grasp_diagnostic_reason = "subject_geometry_unobservable"

        if grasp_contact_candidate:
            grasp_state = GraspState.HELD if self._grasp_confirmed else GraspState.CANDIDATE_HELD
            held_object_ref = instance.subject_ref if self._grasp_confirmed else None
            contact_mode = EffectorContactMode.GRASP_CONTACT if self._grasp_confirmed else EffectorContactMode.CONTACT_CANDIDATE
        else:
            grasp_state = (GraspState.UNKNOWN if gripper_commanded_closed and gripper_blocked
                else GraspState.NOT_HELD)
            held_object_ref = None
            contact_mode = (
                EffectorContactMode.PUSH_CONTACT
                if push_mode
                and supported is True and interaction_estimate.status == "motion_supported"
                else EffectorContactMode.CONTACT_CANDIDATE if interaction_estimate.status in ("candidate", "motion_supported")
                else EffectorContactMode.CLEAR if interaction_estimate.status == "clear"
                else EffectorContactMode.UNKNOWN
            )
        skill = self._skill_state(program, execution)
        if (
            skill.skill_kind == "pick"
            and skill.phase == "descend_to_grasp"
            and centroid is not None
            and extent is not None
        ):
            xy_error = hypot(centroid[0] - tcp[0], centroid[1] - tcp[1])
            lower_z, upper_z = grasp_height_band_from_geometry_v1(centroid, extent)
            if xy_error > self._config.grasp_xy_tolerance_m:
                allowed = ("forward", "backward", "left", "right", "hold")
            elif tcp[2] > upper_z:
                allowed = ("down",)
            elif tcp[2] < lower_z:
                allowed = ("up",)
            else:
                allowed = ("hold",)
            skill = replace(skill, allowed_actions=allowed)
        target_ref = instance.target_ref
        interaction_vector, reference_ref, target_relation = self._interaction(
            skill,
            subject,
            self._reference(visual_scene, target_ref),
            tcp,
        )
        visual_age = max(0.0, raw.simulation_time_s - visual_scene.capture_timestamp_s)
        return StateSnapshotV2(
            state_id=raw.state_id,
            action_epoch=raw.action_epoch,
            timestamp_s=raw.simulation_time_s,
            sources=StateSourcesV2(
                visual_scene_id=visual_scene.scene_id,
                visual_observation_id=visual_scene.observation_id,
                visual_capture_timestamp_s=visual_scene.capture_timestamp_s,
                robot_measurement_timestamp_s=raw.simulation_time_s,
                task_plan_id=task_plan.task_id,
                task_registry_version=TASK_REGISTRY_VERSION,
            ),
            task=self._task_state(task_plan, program, execution, instance.subject_ref, target_ref),
            skill=skill,
            robot=RobotStateV2(
                joint_positions_rad=raw.joint_positions_rad,
                joint_velocities_rad_s=raw.joint_velocities_rad_s,
                tcp_position_robot_base_m=tcp,
                tcp_orientation_robot_base_wxyz=raw.tcp_orientation_robot_base_wxyz,
                gripper=GripperStateV2(
                    opening_m=max(0.0, raw.gripper_joint_m),
                    target_opening_m=max(0.0, raw.gripper_target_m),
                    motion_state=GripperMotionState.STOPPED,
                    right_joint_position_m=raw.right_gripper_joint_m,
                ),
                execution_state=RobotExecutionState.IDLE,
                tool_geometry=tool,
            ),
            physical=PhysicalStateV2(
                grasp_state=grasp_state,
                held_object_ref=held_object_ref,
                effector_contact=EffectorObjectContactV2(
                    subject_ref=(
                        instance.subject_ref
                        if contact_mode in (EffectorContactMode.CONTACT_CANDIDATE,
                            EffectorContactMode.GRASP_CONTACT, EffectorContactMode.PUSH_CONTACT)
                        else None
                    ),
                    mode=contact_mode,
                    left_finger_contact=False if contact_mode is EffectorContactMode.CLEAR else None,
                    right_finger_contact=False if contact_mode is EffectorContactMode.CLEAR else None,
                    support_contact=supported,
                ),
                support_state=SupportStateV2(
                    subject_ref=instance.subject_ref,
                    support_region_ref=visual_support_ref if supported else None,
                    supported=supported,
                ),
                evidence=PhysicalEvidenceV2(
                    robot_feedback_available=True,
                    visual_follow_confirmed=self._visual_follow_confirmed and grasp_contact_candidate,
                    fresh_visual_required=True,
                    grasp_probe_lifted=self._probe_lifted_at_s is not None and grasp_contact_candidate,
                    grasp_probe_failed=self._grasp_probe_failed,
                    contact_source="rgbd_robot_motion_estimate",
                    grasp_source="rgbd_robot_gripper_and_follow_estimate",
                    support_source="rgbd_geometry_support_estimate",
                    grasp_diagnostic=self._grasp_diagnostic(
                        instance.subject_ref, subject, tcp, raw, visual_scene.observation_id),
                    interaction_estimate=interaction_estimate,
                ),
            ),
            visual_scene=visual_scene,
            interaction=InteractionStateV2(
                subject_ref=instance.subject_ref,
                reference_ref=reference_ref,
                local_goal=LocalGoalV2(
                    goal_type=skill.phase,
                    target_relation=target_relation,
                ) if interaction_vector is not None else None,
                derived=DerivedInteractionV2(
                    subject_from_tcp_robot_base_m=interaction_vector,
                    distance_m=_norm(interaction_vector),
                    dominant_axis=_dominant_axis(interaction_vector),
                ) if interaction_vector is not None else None,
            ),
            freshness=FreshnessStateV2(
                visual_age_s=visual_age,
                visual_is_fresh=current_visual,
                prediction_steps=0,
                pending_contact_verification=skill.phase in ("close_gripper", "verify_grasp"),
                pending_completion_verification=skill.status is SkillRunStatus.AWAITING_VERIFICATION,
            ),
            recent_action=self._action_feedback(recent_action),
            decision_feedback=decision_feedback,
        )

    def _reset_grasp_probe(self) -> None:
        self._candidate_tcp = None
        self._candidate_centroid = None
        self._probe_lifted_at_s = None
        self._probe_relative = None
        self._grasp_confirmed = False
        self._grasp_probe_failed = False
        self._visual_follow_confirmed = False
        self._grasp_diagnostic_reason = "contact_candidate_absent"

    def reset_for_retry(self) -> None:
        self._previous_subject_ref = None
        self._reset_grasp_probe()
        self._last_grasp_observation = None
        self._interaction_estimator.reset()

    def _maximum_grasp_relative_drift(self, object_height_m: float | None) -> float:
        maximum = self._config.grasp_maximum_relative_drift_m
        return maximum if object_height_m is None else min(maximum, 0.6 * object_height_m)

    def _grasp_diagnostic(self, subject_ref, subject, tcp, raw, observation_id):
        centroid = subject.geometry.centroid_robot_base_m.value
        extent = subject.geometry.extent_m.value
        relative = None if centroid is None else _subtract(centroid, tcp)
        drift = (None if relative is None or self._probe_relative is None else
            _subtract(relative, self._probe_relative))
        return GraspDiagnosticV1(
            subject_ref, self._grasp_diagnostic_reason, raw.state_id, raw.action_epoch,
            observation_id, raw.simulation_time_s, subject.visibility.value,
            current_relative_robot_base_m=relative,
            candidate_anchor_relative_robot_base_m=(None if self._candidate_centroid is None
                or self._candidate_tcp is None else _subtract(self._candidate_centroid, self._candidate_tcp)),
            probe_anchor_relative_robot_base_m=self._probe_relative,
            relative_drift_robot_base_m=drift,
            relative_drift_m=None if drift is None else _norm(drift),
            maximum_relative_drift_m=self._maximum_grasp_relative_drift(
                None if extent is None else extent[2]),
        )

    def _update_grasp_probe(
        self, centroid: Vec3 | None, tcp: Vec3, timestamp_s: float,
        grasp_contact_candidate: bool, object_height_m: float | None,
        placing_on_support: bool,
    ) -> None:
        if not grasp_contact_candidate or centroid is None:
            self._reset_grasp_probe()
            if grasp_contact_candidate:
                self._grasp_diagnostic_reason = "subject_geometry_unobservable"
            return
        relative = _subtract(centroid, tcp)
        maximum_drift = self._maximum_grasp_relative_drift(object_height_m)
        if self._candidate_tcp is None or self._candidate_centroid is None:
            self._candidate_tcp = tcp
            self._candidate_centroid = centroid
            self._grasp_diagnostic_reason = "awaiting_coupled_lift"
            return
        candidate_relative = _subtract(self._candidate_centroid, self._candidate_tcp)
        tcp_lift = tcp[2] - self._candidate_tcp[2]
        object_lift = centroid[2] - self._candidate_centroid[2]
        if self._probe_lifted_at_s is None:
            if (
                tcp_lift >= self._config.grasp_probe_minimum_lift_m
                and object_lift >= self._config.grasp_probe_minimum_lift_m
                and _norm(_subtract(relative, candidate_relative))
                <= self._config.grasp_probe_maximum_relative_error_m
            ):
                self._probe_lifted_at_s = timestamp_s
                self._probe_relative = relative
                self._visual_follow_confirmed = True
                self._grasp_diagnostic_reason = "awaiting_stable_following"
            elif tcp_lift >= self._config.grasp_probe_maximum_lift_m:
                self._grasp_probe_failed = True
                self._grasp_diagnostic_reason = "coupled_lift_not_confirmed"
            return
        if self._probe_relative is None:
            raise RuntimeError("lifted grasp probe has no relative-pose anchor")
        if not self._grasp_confirmed and _norm(_subtract(relative, self._probe_relative)) > self._config.grasp_probe_maximum_relative_error_m:
            if not self._grasp_probe_failed:
                self._grasp_diagnostic_reason = "probe_relative_drift_exceeded"
            self._grasp_probe_failed = True
        if self._grasp_confirmed and not placing_on_support and _norm(_subtract(relative, self._probe_relative)) > maximum_drift:
            self._grasp_confirmed = False
            self._visual_follow_confirmed = False
            self._grasp_probe_failed = True
            self._grasp_diagnostic_reason = "held_relative_drift_exceeded"
            return
        if self._grasp_probe_failed:
            return
        if timestamp_s - self._probe_lifted_at_s >= self._config.grasp_probe_stable_hold_s:
            self._grasp_confirmed = True
            self._grasp_diagnostic_reason = "held_confirmed"

    @staticmethod
    def _current_instance(program: CapabilityProgramV1, execution: CapabilityExecutionStateV1):
        if execution.active_step_index is not None:
            return program.steps[execution.active_step_index]
        pending = next(
            (step for index, step in enumerate(program.steps) if execution.steps[index].status is CapabilityRunStatus.PENDING),
            None,
        )
        return pending or program.steps[-1]

    @staticmethod
    def _skill_state(
        program: CapabilityProgramV1, execution: CapabilityExecutionStateV1
    ) -> SkillStateV2:
        if execution.active_step_index is not None:
            return active_skill_state_v2(program, execution)
        instance = LiveStateSnapshotBuilderV2._current_instance(program, execution)
        status = (
            SkillRunStatus.COMPLETED
            if execution.status is CapabilityProgramStatus.COMPLETED
            else SkillRunStatus.READY
        )
        return SkillStateV2(
            skill_instance_id=instance.instance_id,
            skill_kind=instance.kind.value,
            phase="completed" if status is SkillRunStatus.COMPLETED else "entry_gate",
            subject_ref=instance.subject_ref,
            target_ref=instance.target_ref,
            source_ref=instance.source_ref,
            allowed_intents=(),
            allowed_actions=(),
            status=status,
            goal=(
                None
                if instance.goal is None
                else SkillGoalV1(
                    predicate=instance.goal.predicate,
                    reference_ref=instance.goal.reference_ref,
                    desired_value=instance.goal.desired_value,
                )
            ),
        )

    @staticmethod
    def _task_state(
        plan: TaskPlanV1,
        program: CapabilityProgramV1,
        execution: CapabilityExecutionStateV1,
        subject_ref: str,
        target_ref: str | None,
    ) -> TaskStateV2:
        status = {
            CapabilityProgramStatus.READY: TaskRunStatus.READY,
            CapabilityProgramStatus.RUNNING: TaskRunStatus.RUNNING,
            CapabilityProgramStatus.COMPLETED: TaskRunStatus.COMPLETED,
            CapabilityProgramStatus.BLOCKED: TaskRunStatus.BLOCKED,
        }[execution.status]
        subject_refs = tuple(dict.fromkeys(step.subject_ref for step in program.steps))
        completed = tuple(
            candidate
            for candidate in subject_refs
            if all(
                execution.steps[index].status is CapabilityRunStatus.COMPLETED
                for index, step in enumerate(program.steps)
                if step.subject_ref == candidate
            )
        )
        pending = tuple(candidate for candidate in subject_refs if candidate not in completed)
        return TaskStateV2(
            task_id=plan.task_id,
            task_kind=plan.task.kind,
            status=status,
            completed_subject_refs=completed,
            pending_subject_refs=pending,
            current_subject_ref=subject_ref,
            current_goal_ref=target_ref,
        )

    @staticmethod
    def _interaction(
        skill: SkillStateV2,
        subject: EntityV1,
        target: RegionV1 | EntityV1 | None,
        tcp: Vec3,
    ) -> tuple[Vec3 | None, str | None, str]:
        # This field is an observation of subject relative to TCP, never an
        # overloaded final-goal error. Phase-specific corrections are projected
        # separately at the policy boundary from named public measurements.
        del skill, target
        centroid = subject.geometry.centroid_robot_base_m.value
        if centroid is None or subject.visibility is not Visibility.CLEAR:
            return None, subject.track_id, "unobservable"
        return _subtract(centroid, tcp), subject.track_id, "tcp_to_subject"

    @staticmethod
    def _entity(scene: VisualSceneV1, reference: str) -> EntityV1:
        return next(entity for entity in scene.entities if entity.track_id == reference)

    @staticmethod
    def _region(scene: VisualSceneV1, reference: str | None) -> RegionV1 | None:
        if reference is None:
            return None
        return next((region for region in scene.regions if region.region_id == reference), None)

    @staticmethod
    def _reference(
        scene: VisualSceneV1, reference: str | None
    ) -> RegionV1 | EntityV1 | None:
        region = LiveStateSnapshotBuilderV2._region(scene, reference)
        if region is not None or reference is None:
            return region
        return next(
            (entity for entity in scene.entities if entity.track_id == reference),
            None,
        )

    @staticmethod
    def _action_feedback(result: ActionResult | None) -> ActionFeedbackV1 | None:
        if result is None or result.end_action_epoch <= result.start_action_epoch:
            return None
        return ActionFeedbackV1(
            decision_id=result.decision_id,
            action=result.action.value,
            start_action_epoch=result.start_action_epoch,
            end_action_epoch=result.end_action_epoch,
            status=result.status.value,
            commanded_delta_robot_base_m=result.commanded_delta_robot_base_m,
            measured_delta_robot_base_m=result.measured_delta_robot_base_m,
            end_timestamp_s=result.end_simulation_time_s,
        )
