"""从公开几何提供候选，保存模型选择，按新观测检查意图有效性。"""

from dataclasses import dataclass, replace
from math import hypot, isfinite
from pathlib import Path
import yaml

from jev4mujoco.contracts.local_intent import (
    LocalIntentCandidateV1, LocalIntentChoiceV1, LocalIntentConditionV1, LocalIntentContextV1, LocalIntentExecutionV1,
    LocalIntentHandoffV1, LocalIntentTerminationV1, LocalIntentIntervalV1,
    PlacementCorrectionV1, PushContactPlanV1, PushPreparationV1, PushGoalExecutionV1,
)
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2, PhaseConditionV1, PhaseObservationBasisV1, GraspState
from jev4mujoco.contracts.visual_scene import Visibility
from jev4mujoco.perception.surface_geometry import signed_point_polygon_clearance_v1
from jev4mujoco.planning.phase_definitions import bound_subtask_definition_v1
from jev4mujoco.planning.capabilities import CAPABILITY_TEMPLATES_V1, CapabilityKind
from jev4mujoco.runtime.grasp_geometry import grasp_height_band_v1
from jev4mujoco.runtime.grasp_observation import (
    grasp_recovery_mode_v1, grasp_recovery_candidates_v1, grasp_observation_motion_ready_v1,
    grasp_preparation_ready_v1, current_grasp_approach_v1, grasp_preparation_goal_reached_v1,
)
from jev4mujoco.runtime.tool_geometry import observed_contact_sides_v1, contact_geometry_v1
from jev4mujoco.runtime.push_interaction import (push_interaction_mode_v1, current_push_plan_side_v1,
    push_reposition_ready_v1, push_outer_side_point_v1, push_side_aligned_v1)
from jev4mujoco.runtime.push_goal_plan import (
    native_push_goal_observation_v1, native_push_goal_improved_v1, update_best_exit_progress_v1, push_plan_step_v1,
)
from jev4mujoco.runtime.placement_preparation import (
    placement_preparation_candidates_v1, placement_preparation_comparable_v1,
    placement_preparation_goal_reached_v1,
)


@dataclass(frozen=True, slots=True)
class LocalIntentConfigV1:
    minimum_progress_m: float
    maximum_actions_without_new_best: int
    maximum_local_replans: int
    maximum_reference_shift_m: float

    def __post_init__(self):
        for value in (self.minimum_progress_m, self.maximum_reference_shift_m):
            if not isfinite(value) or value <= 0:
                raise ValueError("local intent distances must be finite and positive")
        for value in (self.maximum_actions_without_new_best, self.maximum_local_replans):
            if type(value) is not int or value <= 0:
                raise ValueError("local intent limits must be positive integers")


def load_local_intent_config_v1(path: str | Path) -> LocalIntentConfigV1:
    data = yaml.safe_load(Path(path).read_text())
    if data.pop("schema", None) != "LocalIntentConfigV1":
        raise ValueError("unsupported local intent configuration")
    return LocalIntentConfigV1(**data)


def local_intent_metric_v1(candidate, snapshot):
    if candidate.placement_preparation is not None and not placement_preparation_comparable_v1(candidate, snapshot):
        return None
    if candidate.choice_id == "observe_grasp_geometry":
        return current_runtime_measurements_v1(snapshot).get("grasp_observation_span_deficit_m")
    if candidate.choice_id.startswith("prepare_grasp_view_") and not grasp_preparation_ready_v1(snapshot):
        return None
    if candidate.choice_id.startswith("prepare_push_side_edge_"):
        values = current_runtime_measurements_v1(snapshot)
        height, clearance = values.get("push_reposition_height_deficit_m"), values.get("push_tool_above_subject_clearance_m")
        return None if height is None or clearance is None else max(height, -clearance, 0.0)
    if candidate.choice_id.startswith(("restore_push_contact_edge_", "contact_reprepared_push_edge_")):
        values = current_runtime_measurements_v1(snapshot)
        gap = contact_geometry_v1(snapshot, candidate.contact_side)["local_error_m"]
        height = values.get("push_height_deficit_m")
        return None if gap is None or height is None else max(float(gap), height)
    if candidate.choice_id == "observe_task_geometry" and candidate.metric_kind == "runtime_conditions":
        return current_runtime_measurements_v1(snapshot).get("task_observation_span_deficit_m")
    if candidate.choice_id == "retract_after_release" and candidate.metric_kind == "runtime_conditions":
        return (current_runtime_measurements_v1(snapshot).get("tool_transfer_clearance_deficit_m")
            if tool_retraction_motion_ready_v1(snapshot) else None)
    if candidate.progress_interval is not None:
        interval = candidate.progress_interval
        value = local_intent_interval_value_v1(interval, snapshot)
        if (candidate.reference_ref != snapshot.skill.target_ref
            or (candidate.choice_id in ("lower_to_release", "approach_placement_support")
                and placement_relation_restore_mode_v1(snapshot) != "placement")):
            return None
        return None if value is None else local_intent_interval_distance_v1(interval, value)
    target = candidate.target_robot_base_m
    tcp = snapshot.robot.tcp_position_robot_base_m
    if candidate.metric_kind == "tcp_xy":
        return hypot(target[0] - tcp[0], target[1] - tcp[1])
    if candidate.metric_kind == "tcp_z":
        return abs(target[2] - tcp[2])
    if candidate.metric_kind == "contact_geometry":
        return contact_geometry_v1(snapshot, candidate.contact_side)["local_error_m"]
    return None  # A Boolean phase result is not a continuous progress estimate.


def local_intent_correction_v1(candidate, snapshot):
    if candidate.placement_preparation is not None and not placement_preparation_comparable_v1(candidate, snapshot):
        return None
    if candidate.choice_id == "observe_grasp_geometry":
        r=current_grasp_approach_v1(snapshot)
        return (None if r is None or not grasp_observation_motion_ready_v1(snapshot) else
            (0.,0.,max(0.,r.return_observation.tcp_robot_base_m[2]+r.retraction_margin_m-snapshot.robot.tcp_position_robot_base_m[2])))
    if candidate.choice_id.startswith("prepare_push_side_edge_"):
        delta = current_runtime_measurements_v1(snapshot).get("tcp_to_push_reposition_height_z_m")
        return None if delta is None else (0.0, 0.0, delta)
    if candidate.choice_id == "retract_after_release" and candidate.metric_kind == "runtime_conditions":
        deficit = local_intent_metric_v1(candidate, snapshot)
        return None if deficit is None else (0.0, 0.0, deficit)
    if candidate.progress_interval is not None:
        metric = local_intent_metric_v1(candidate, snapshot)
        value = local_intent_interval_value_v1(candidate.progress_interval, snapshot)
        if metric is None or value is None:
            return None
        interval = candidate.progress_interval
        return (0.0, 0.0, min(interval.maximum_m, max(interval.minimum_m, value)) - value)
    target = candidate.target_robot_base_m
    tcp = snapshot.robot.tcp_position_robot_base_m
    if candidate.metric_kind == "tcp_xy":
        return (target[0] - tcp[0], target[1] - tcp[1], 0.0)
    if candidate.metric_kind == "tcp_z":
        return (0.0, 0.0, target[2] - tcp[2])
    if candidate.metric_kind == "contact_geometry" or candidate.choice_id.startswith(("restore_push_contact_edge_", "contact_reprepared_push_edge_")):
        return contact_geometry_v1(snapshot, candidate.contact_side).get("correction_robot_base_m")
    return None


def current_runtime_measurements_v1(snapshot):
    """仅使用同绑定、同观测且动作后的 Runtime 测量，旧反馈不能授权动作。"""
    feedback = snapshot.phase_feedback
    basis = None if feedback is None else feedback.observation_basis
    evidence = None if feedback is None else feedback.evidence_basis
    action_end = 0.0 if snapshot.recent_action is None else snapshot.recent_action.end_timestamp_s
    if (feedback is None or basis is None or evidence is None
        or (feedback.based_on_state_id, feedback.based_on_action_epoch, feedback.based_on_observation_id,
            feedback.skill_instance_id, feedback.phase)
        != (snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id,
            snapshot.skill.skill_instance_id, snapshot.skill.phase)
        or (basis.state_id, basis.action_epoch, basis.observation_id)
        != (snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id)
        or not PhaseObservationBasisV1.same_scope(feedback.based_on_skill, snapshot.skill)
        or not basis.same_scope_as(snapshot.skill)
        or not snapshot.freshness.visual_is_fresh
        or not evidence.visual_evidence_current or not evidence.robot_evidence_current
        or basis.evidence_basis != evidence
        or evidence.visual_capture_timestamp_s != snapshot.sources.visual_capture_timestamp_s
        or evidence.visual_capture_timestamp_s != snapshot.visual_scene.capture_timestamp_s
        or evidence.robot_measurement_timestamp_s != snapshot.sources.robot_measurement_timestamp_s
        or evidence.robot_measurement_timestamp_s != snapshot.timestamp_s
        or min(evidence.visual_capture_timestamp_s, evidence.robot_measurement_timestamp_s) < action_end):
        return {}
    validity = dict(basis.measurement_evidence_current)
    return {item.name: item.value for item in feedback.measurements if validity.get(item.name) is True}


def local_intent_conditions_v1(candidate, snapshot):
    values = current_runtime_measurements_v1(snapshot)
    return tuple(PhaseConditionV1(spec.measurement, values.get(spec.measurement), spec.operator,
        values.get(spec.required) if isinstance(spec.required, str) else spec.required,
        evidence_current=spec.measurement in values
            and (not isinstance(spec.required, str) or spec.required in values))
        for spec in candidate.completion_conditions)


def local_intent_condition_status_v1(candidate, snapshot):
    if candidate.progress_interval is not None and local_intent_metric_v1(candidate, snapshot) is None:
        return None
    conditions = local_intent_conditions_v1(candidate, snapshot)
    results = tuple(item.satisfied for item in conditions)
    return None if not results or any(item is None for item in results) else all(results)


def local_intent_interval_value_v1(interval, snapshot):
    value = current_runtime_measurements_v1(snapshot).get(interval.measurement)
    measurement = next((m for m in snapshot.phase_feedback.measurements if m.name == interval.measurement), None) if snapshot.phase_feedback else None
    return value if (measurement is not None and measurement.unit == "m"
        and type(value) in (float, int)) else None


def local_intent_interval_distance_v1(interval, value):
    return max(0.0, interval.minimum_m - value, value - interval.maximum_m)


def placement_height_interval_v1(snapshot, *, maximum_measurement="placement_height_adjustment_maximum_clearance_m"):
    if placement_relation_restore_mode_v1(snapshot) != "placement":
        return None
    maximum = current_runtime_measurements_v1(snapshot).get(maximum_measurement)
    measurement = next((m for m in snapshot.phase_feedback.measurements
        if m.name == maximum_measurement), None)
    if (measurement is None or measurement.unit != "m" or measurement.source != "deployment_configuration"
        or type(maximum) not in (float, int) or maximum <= 0):
        return None
    return LocalIntentIntervalV1("subject_bottom_to_target_support_clearance_m", 0.0, maximum)


class PlacementCorrectionRuntimeV1:
    """两种JEV模式共用的纠正流程；记录义务，不选择动作。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._correction = None

    def attach(self, snapshot):
        skill = snapshot.skill
        if skill.skill_kind != "place" or skill.phase != "lower_to_release":
            self.reset()
            return replace(snapshot, placement_correction=None)
        if self._correction is not None and not self._correction.matches(skill):
            self.reset()
        values = current_runtime_measurements_v1(snapshot)
        if snapshot.phase_feedback is not None and snapshot.phase_feedback.outcome in ("passed", "failed"):
            self.reset()
        elif values.get("grasp_not_held") is True and snapshot.physical.grasp_state is GraspState.NOT_HELD:
            self.reset()
        elif (values.get("subject_held") is True and snapshot.physical.grasp_state is GraspState.HELD
            and snapshot.physical.held_object_ref == skill.subject_ref):
            clearance, required = values.get("subject_bottom_clearance_m"), values.get("required_bottom_clearance_m")
            if clearance is not None and required is not None:
                if (values.get("placement_horizontal_relation_satisfied") is False and clearance < required
                    and (self._correction is None or self._correction.status == "completed")):
                    goal = None if skill.goal is None else (skill.goal.predicate.value, skill.goal.reference_ref, skill.goal.desired_value)
                    self._correction = PlacementCorrectionV1(skill.skill_instance_id, skill.phase,
                        skill.subject_ref, skill.source_ref, skill.target_ref, goal, snapshot.state_id,
                        snapshot.action_epoch, snapshot.visual_scene.observation_id)
                if self._correction is not None and self._correction.status == "pending_clearance" and clearance >= required:
                    self._correction = replace(self._correction, status="completed", resolved_state_id=snapshot.state_id,
                        resolved_action_epoch=snapshot.action_epoch, resolved_observation_id=snapshot.visual_scene.observation_id)
        return replace(snapshot, placement_correction=self._correction)


def placement_relation_restore_mode_v1(snapshot):
    if snapshot.skill.skill_kind != "place" or snapshot.skill.phase != "lower_to_release":
        return None
    feedback = snapshot.phase_feedback
    values = (current_runtime_measurements_v1(snapshot) if feedback is not None
        and feedback.contract_id == "release_candidate_v1" else {})
    held = (values.get("subject_held") is True
        and snapshot.physical.grasp_state is GraspState.HELD
        and snapshot.physical.held_object_ref == snapshot.skill.subject_ref)
    if not held:
        return ("released" if values.get("grasp_not_held") is True
            and snapshot.physical.grasp_state is GraspState.NOT_HELD else "unknown")
    relation = values.get("placement_horizontal_relation_satisfied")
    correction = snapshot.placement_correction
    if correction is not None and correction.matches(snapshot.skill) and correction.status == "pending_clearance":
        clearance, required = values.get("subject_bottom_clearance_m"), values.get("required_bottom_clearance_m")
        if relation is None or clearance is None or required is None:
            return "unknown"
        if clearance < required:
            return "restore_clearance"
    if relation is True:
        return "placement"
    clearance, required = values.get("subject_bottom_clearance_m"), values.get("required_bottom_clearance_m")
    if relation is not False or clearance is None or required is None:
        return "unknown"
    return "restore_clearance" if clearance < required else "restore_relation"


def placement_horizontal_preparation_allowed_v1(snapshot):
    if placement_relation_restore_mode_v1(snapshot) != "placement":
        return False
    values = current_runtime_measurements_v1(snapshot)
    clearance, required = values.get("subject_bottom_clearance_m"), values.get("required_bottom_clearance_m")
    return clearance is not None and required is not None and clearance >= required


def tool_retraction_motion_ready_v1(snapshot):
    feedback = snapshot.phase_feedback
    if (snapshot.skill.skill_kind != "place" or snapshot.skill.phase != "retract_after_release"
        or feedback is None or feedback.contract_id != "tool_transfer_clearance_v1"):
        return False
    values = current_runtime_measurements_v1(snapshot)
    return (all(values.get(name) is True for name in (
        "grasp_not_held", "gripper_open_observed", "subject_supported", "capability_goal_satisfied"))
        and values.get("tool_transfer_clearance_m") is not None
        and values.get("tool_transfer_clearance_deficit_m") is not None
        and values.get("tool_transfer_required_clearance_m") is not None)


def task_observation_motion_ready_v1(snapshot):
    feedback = snapshot.phase_feedback
    if (snapshot.skill.skill_kind != "place" or snapshot.skill.phase != "observe_task_geometry"
        or feedback is None or feedback.contract_id != "released_goal_observation_v1"):
        return False
    values = current_runtime_measurements_v1(snapshot)
    clearance, required = values.get("tool_transfer_clearance_m"), values.get("tool_transfer_required_clearance_m")
    return (all(values.get(name) is True for name in (
        "grasp_not_held", "gripper_open_observed", "subject_supported", "capability_goal_satisfied"))
        and clearance is not None and required is not None and clearance >= required
        and values.get("required_task_geometry_observed") is False
        and values.get("task_observation_span_deficit_m") is not None
        and values.get("task_observation_robot_occlusion_hint") is True)


def apply_placement_action_permissions_v1(snapshot):
    """阶段模板描述潜在能力；当前证据收窄权限，供一层和两层共同使用。"""
    grasp_mode=grasp_recovery_mode_v1(snapshot)
    if grasp_mode is not None:
        if grasp_mode == "observe":
            allowed=("up","hold") if grasp_observation_motion_ready_v1(snapshot) else ("hold",)
        elif grasp_mode == "unknown" or not grasp_preparation_ready_v1(snapshot):
            allowed=("hold",)
        elif grasp_mode == "prepare":
            allowed=("forward","backward","left","right","hold")
        else:
            allowed=("forward","backward","left","right","up","down","hold")
        skill=replace(snapshot.skill,allowed_actions=allowed)
        feedback=snapshot.phase_feedback
        if feedback is not None:
            feedback=replace(feedback,based_on_skill=skill,observation_basis=None if feedback.observation_basis is None
                else replace(feedback.observation_basis,skill=skill))
        return replace(snapshot,skill=skill,phase_feedback=feedback)
    push_mode = push_interaction_mode_v1(snapshot)
    if push_mode is not None:
        phase = next(item for item in CAPABILITY_TEMPLATES_V1[CapabilityKind.SURFACE_PUSH].phases
            if item.phase_id == snapshot.skill.phase)
        allowed = ("hold",) if push_mode in ("unknown", "blocked") else tuple(action.value for action in phase.allowed_actions)
        preparation = None if snapshot.local_intent is None else snapshot.local_intent.push_preparation
        if preparation is not None and push_mode not in ("unknown", "blocked"):
            if push_reposition_ready_v1(snapshot) is None:
                allowed = ("hold",)
            elif preparation.stage == "clearance" or (preparation.stage == "align_side" and push_reposition_ready_v1(snapshot) is not True):
                allowed = ("up", "down", "hold")
            elif preparation.stage == "align_side":
                allowed = ("forward", "backward", "left", "right", "hold")
        skill = replace(snapshot.skill, allowed_actions=allowed)
        feedback = snapshot.phase_feedback
        if feedback is not None:
            feedback = replace(feedback, based_on_skill=skill,
                observation_basis=None if feedback.observation_basis is None else
                    replace(feedback.observation_basis, skill=skill))
        return replace(snapshot, skill=skill, phase_feedback=feedback)
    mode = placement_relation_restore_mode_v1(snapshot)
    if snapshot.skill.skill_kind == "place" and snapshot.skill.phase == "retract_after_release":
        allowed = ("up", "hold") if tool_retraction_motion_ready_v1(snapshot) else ("hold",)
    elif snapshot.skill.skill_kind == "place" and snapshot.skill.phase == "observe_task_geometry":
        allowed = ("up", "hold") if task_observation_motion_ready_v1(snapshot) else ("hold",)
    elif mode is None:
        return snapshot
    else:
        allowed = {
            "placement": ("down", "up", "hold", "gripper_open"),
            "released": ("gripper_open", "hold"),
            "restore_clearance": ("up", "hold"),
            "restore_relation": ("forward", "backward", "left", "right", "hold"),
            "unknown": ("hold",),
        }[mode]
    if mode == "placement" and placement_horizontal_preparation_allowed_v1(snapshot):
        allowed += ("forward", "backward", "left", "right")
    # 从模板重新求交，避免沿用上次动态许可而无法切换纠正模式。
    phase = next(item for item in CAPABILITY_TEMPLATES_V1[CapabilityKind.PLACE].phases
        if item.phase_id == snapshot.skill.phase)
    actions = tuple(action.value for action in phase.allowed_actions if action.value in allowed)
    skill = replace(snapshot.skill, allowed_actions=actions)
    feedback = snapshot.phase_feedback
    if feedback is not None:
        # 同次证据不变；当前权限投影同步绑定，历史 observation_basis 不刷新。
        feedback = replace(feedback, based_on_skill=skill,
            observation_basis=(None if feedback.observation_basis is None else
                replace(feedback.observation_basis, skill=skill)))
    return replace(snapshot, skill=skill, phase_feedback=feedback)


def local_intent_candidates_v1(snapshot: StateSnapshotV2, *, for_validation=False, retained_approach=None,
                              push_contact_plan=None, push_preparation=None, placement_preparation=None,
                              placement_preparation_step_m=None, placement_preparation_tolerance_m=None,
                              placement_preparation_minimum_improvement_m=None) -> tuple[LocalIntentCandidateV1, ...]:
    """Candidates specify local conditions; their order is not a route ranking."""
    skill, feedback = snapshot.skill, snapshot.phase_feedback
    if feedback is None or (not skill.allowed_actions and not for_validation):
        return ()
    definition = bound_subtask_definition_v1(skill.skill_instance_id, skill.skill_kind,
        skill.subject_ref, skill.source_ref, skill.target_ref, None if skill.goal is None else
        (skill.goal.predicate.value, skill.goal.reference_ref, skill.goal.desired_value))
    phase = next(item[2] for item in definition.phases if item[0] == skill.phase)
    parameters = dict(feedback.parameters)
    measurements = {item.name: item.value for item in feedback.measurements}
    entity = next((item for item in snapshot.visual_scene.entities if item.track_id == skill.subject_ref), None)
    target = next((item for item in snapshot.visual_scene.regions if item.region_id == skill.target_ref), None)
    tcp = snapshot.robot.tcp_position_robot_base_m
    candidates = []

    def geometric(name, objective, axes, point, tolerance, reference, side=None):
        # Geometry remains observable when Runtime changes the permitted action axes.
        available = ("forward", "backward", "left", "right", "up", "down", "hold") if for_validation else skill.allowed_actions
        actions = tuple(action for action in available if action == "hold" or
            action in ({"forward", "backward", "left", "right"} if axes == "xy" else {"up", "down"}))
        if actions:
            candidate = LocalIntentCandidateV1(name, objective, actions, "tcp_" + axes,
                point, tolerance, reference, phase.maintain, "public_rgbd_and_robot_feedback", contact_side=side)
            candidates.append(candidate)

    if feedback.contract_id == "tcp_subject_xy_aligned_v1":
        point = None if entity is None or entity.visibility is not Visibility.CLEAR else entity.geometry.top_center_robot_base_m.value
        if point is not None:
            geometric("align_subject_xy", phase.objective, "xy", point, parameters["grasp_xy_tolerance_m"], skill.subject_ref)
    elif feedback.contract_id == "tcp_grasp_pose_aligned_v1":
        extra=grasp_recovery_candidates_v1(snapshot,for_validation=for_validation)
        candidates.extend(extra)
        point = None if entity is None or entity.visibility is not Visibility.CLEAR else entity.geometry.centroid_robot_base_m.value
        band = grasp_height_band_v1(snapshot, skill.subject_ref)
        if point is not None and band is not None and (for_validation
            or grasp_recovery_mode_v1(snapshot) not in ("observe","unknown","prepare")):
            geometric("align_grasp_xy", "Align TCP with subject centroid in XY, preserving height.", "xy", point,
                parameters["grasp_xy_tolerance_m"], skill.subject_ref)
            if hypot(point[0] - tcp[0], point[1] - tcp[1]) <= parameters["grasp_xy_tolerance_m"]:
                geometric("adjust_grasp_height", "Reach the observed grasp-height band while preserving XY alignment.", "z",
                    (tcp[0], tcp[1], 0.5 * (band[0] + band[1])), 0.5 * (band[1] - band[0]), skill.subject_ref)
    elif feedback.contract_id == "press_point_aligned_v1":
        point = None if target is None or target.geometry.point_robot_base_m is None else target.geometry.point_robot_base_m.value
        if point is not None:
            geometric("align_press_point", phase.objective, "xy", point, parameters["press_point_xy_tolerance_m"], skill.target_ref)
    elif feedback.contract_id == "transport_clearance_v1":
        deficit = measurements.get("subject_bottom_clearance_deficit_m")
        if deficit is not None and deficit > 0:
            candidates.append(LocalIntentCandidateV1("raise_subject_clearance", phase.objective,
                tuple(action for action in skill.allowed_actions if action in ("up", "down", "hold")),
                "phase_contract", reference_ref=skill.subject_ref, maintain=phase.maintain))
    elif feedback.contract_id == "tool_transfer_clearance_v1":
        actions = tuple(action for action in skill.allowed_actions if action in ("up", "hold"))
        if actions:
            candidates.append(LocalIntentCandidateV1("retract_after_release", phase.objective,
                actions, "runtime_conditions", reference_ref=skill.subject_ref, maintain=phase.maintain,
                completion_conditions=tuple(LocalIntentConditionV1(name, "equals", True) for name in (
                    "grasp_not_held", "gripper_open_observed", "subject_supported", "capability_goal_satisfied"))
                    + (LocalIntentConditionV1("tool_transfer_clearance_m", "at_least", "tool_transfer_required_clearance_m"),)))
    elif feedback.contract_id == "released_goal_observation_v1":
        actions = tuple(action for action in skill.allowed_actions if action in ("up", "hold"))
        if actions:
            candidates.append(LocalIntentCandidateV1("observe_task_geometry", phase.objective,
                actions, "runtime_conditions", reference_ref=skill.subject_ref, maintain=phase.maintain,
                completion_conditions=tuple(LocalIntentConditionV1(name, "equals", True) for name in (
                    "grasp_not_held", "gripper_open_observed", "subject_supported", "capability_goal_satisfied",
                    "required_task_geometry_observed"))
                    + (LocalIntentConditionV1("tool_transfer_clearance_m", "at_least", "tool_transfer_required_clearance_m"),)))
    elif feedback.contract_id == "push_precontact_aligned_v1":
        footprint = None if entity is None or entity.visibility is not Visibility.CLEAR else entity.geometry.footprint_xy_robot_base_m.value
        source = next((item for item in snapshot.visual_scene.regions if item.region_id == skill.source_ref), None)
        boundary = None if source is None or source.geometry.boundary_xy_robot_base_m is None else source.geometry.boundary_xy_robot_base_m.value
        if footprint is not None and boundary is not None:
            sides = {side.edge_index: side for side in observed_contact_sides_v1(snapshot)}
            area2 = sum(footprint[i - 1][0] * point[1] - point[0] * footprint[i - 1][1] for i, point in enumerate(footprint))
            clearance = parameters["surface_push_precontact_clearance_m"]
            tolerance = parameters["surface_push_precontact_xy_tolerance_m"]
            for index, end in enumerate(footprint):
                start = footprint[index - 1]
                dx, dy = end[0] - start[0], end[1] - start[1]
                length = hypot(dx, dy)
                if length <= 1e-12 or index not in sides:
                    continue
                orientation = 1.0 if area2 > 0 else -1.0
                point = ((start[0] + end[0]) * 0.5 + orientation * dy / length * clearance,
                    (start[1] + end[1]) * 0.5 - orientation * dx / length * clearance, tcp[2])
                # Public point geometry only; no arm/tool path safety is inferred.
                if (signed_point_polygon_clearance_v1(point[:2], boundary) <= 0
                    and abs(signed_point_polygon_clearance_v1(point[:2], footprint) - clearance) <= tolerance):
                    geometric(f"approach_edge_{index}", "Approach this observed outer-side point in XY. Cross-phase side inheritance requires fresh verified reach. Preserve tool height while aligning.",
                        "xy", point, tolerance, skill.subject_ref,
                        sides[index])
        height_delta = measurements.get("tcp_to_push_height_z_m")
        if height_delta is not None:
            geometric("adjust_push_height", "Reach the configured tool height relative to source support, preserving TCP XY.", "z",
                (tcp[0], tcp[1], tcp[2] + height_delta), parameters["surface_push_tcp_height_tolerance_m"], skill.source_ref)
    elif feedback.contract_id == "push_contact_established_v1":
        sides = observed_contact_sides_v1(snapshot)
        if retained_approach is not None and retained_approach.contact_side is not None:
            sides = tuple(s for s in sides if s.edge_index == retained_approach.contact_side.edge_index)
        if snapshot.robot.tool_geometry is not None:
            for side in sides:
                candidates.append(LocalIntentCandidateV1(f"contact_edge_{side.edge_index}",
                    "Establish actual supported contact on this selected finite side; use tool gap, tangent and height overlap, and preserve support. Geometric overlap alone does not confirm contact.",
                    (skill.allowed_actions or (("forward", "backward", "left", "right", "up", "down", "hold")
                        if for_validation else ())), "contact_geometry", reference_ref=skill.subject_ref,
                    maintain=phase.maintain, geometry_source="calibrated_tool_and_public_rgbd",
                    contact_side=side))
    elif feedback.contract_id == "surface_push_goal_reached_v1":
        mode = push_interaction_mode_v1(snapshot)
        if mode in ("unknown", "blocked"):
            return (LocalIntentCandidateV1("observe_push_evidence",
                "Wait for current push prerequisites; missing evidence does not permit translation and known failed prerequisites block pushing.",
                ("hold",), "phase_contract", reference_ref=skill.subject_ref),)
        side = current_push_plan_side_v1(push_contact_plan, snapshot)
        if push_preparation is None and mode == "advance":
            candidates.append(LocalIntentCandidateV1("advance_surface_push", phase.objective,
                skill.allowed_actions, "phase_contract", reference_ref=skill.target_ref,
                maintain=phase.maintain, geometry_source="calibrated_tool_and_public_rgbd", contact_side=side))
        elif push_preparation is None:
            for current_side in observed_contact_sides_v1(snapshot):
                candidates.append(LocalIntentCandidateV1(f"restore_push_contact_edge_{current_side.edge_index}",
                    "Restore actual supported subject contact and original TCP height using this current finite side. Contact/height are correction goals, not prerequisites. The model may retain its historical side plan or select another current side; zero geometric distance alone never completes contact.",
                    skill.allowed_actions, "runtime_conditions", reference_ref=skill.subject_ref,
                    maintain=("subject_supported", "subject_not_held", "avoid_other_objects"),
                    geometry_source="calibrated_tool_and_public_rgbd", contact_side=current_side,
                    completion_conditions=tuple(LocalIntentConditionV1(name, "equals", True) for name in (
                        "supported_push_contact", "subject_supported", "grasp_not_held", "push_height_ready"))))
        # 再准备侧面由JEV明确选择，Runtime不按目标关系排序或替模型换面。
        available = tuple(action.value for definition in CAPABILITY_TEMPLATES_V1[CapabilityKind.SURFACE_PUSH].phases
            if definition.phase_id == skill.phase for action in definition.allowed_actions)
        sides = observed_contact_sides_v1(snapshot) if push_preparation is None else tuple(
            s for s in observed_contact_sides_v1(snapshot) if s == current_push_plan_side_v1(push_preparation.plan, snapshot))
        for current_side in sides:
            point = push_outer_side_point_v1(snapshot, current_side)
            if point is None or push_reposition_ready_v1(snapshot) is None:
                continue
            stage = (push_preparation.stage if push_preparation is not None else
                "clearance" if push_reposition_ready_v1(snapshot) is not True else
                "contact" if push_side_aligned_v1(snapshot, current_side) else "align_side")
            if stage == "align_side" and push_reposition_ready_v1(snapshot) is not True:
                stage = "clearance"
            common = dict(reference_ref=skill.subject_ref,
                maintain=("subject_supported", "subject_not_held", "avoid_other_objects"),
                geometry_source="calibrated_tool_and_public_rgbd", contact_side=current_side)
            if stage == "clearance":
                candidates.append(LocalIntentCandidateV1(f"prepare_push_side_edge_{current_side.edge_index}",
                    "Select this current side plan and prepare the original reposition-height band, tool above subject and actual contact clear before XY re-alignment. This changes a local plan, not the final goal or task recovery.",
                    ("up", "down", "hold"), "runtime_conditions", **common,
                    completion_conditions=tuple(LocalIntentConditionV1(name, "equals", True) for name in (
                        "push_reposition_ready", "subject_supported", "grasp_not_held", "push_tool_closed_observed"))))
            elif stage == "align_side":
                candidates.append(LocalIntentCandidateV1(f"align_reprepared_push_edge_{current_side.edge_index}",
                    "Align TCP XY with this selected current outer-side point at the prepared height; maintain current tool clearance and empty closed-tool feedback. The point is rebuilt from each current footprint.",
                    tuple(a for a in available if a in ("forward", "backward", "left", "right", "hold")),
                    "tcp_xy", point, current_runtime_measurements_v1(snapshot)["push_precontact_xy_tolerance_m"], **common))
            else:
                candidates.append(LocalIntentCandidateV1(f"contact_reprepared_push_edge_{current_side.edge_index}",
                    "Restore original push height and actual supported subject contact using this selected current finite side. Preparation history is not contact proof; zero gap never substitutes for original contact verification.",
                    available, "runtime_conditions", **common,
                    completion_conditions=tuple(LocalIntentConditionV1(name, "equals", True) for name in (
                        "supported_push_contact", "subject_supported", "grasp_not_held", "push_height_ready", "push_tool_closed_observed"))))
    elif feedback.contract_id == "release_candidate_v1":
        mode = placement_relation_restore_mode_v1(snapshot)
        recovery = {
            "restore_clearance": ("restore_placement_clearance",
                "Complete the current clearance-correction obligation using fresh measurements, even if horizontal alignment temporarily returns. Reach the existing transport clearance before horizontal preparation; clearance does not certify an arm/tool path.",
                ("up", "hold"), LocalIntentConditionV1("subject_bottom_clearance_m", "at_least", "required_bottom_clearance_m")),
            "restore_relation": ("restore_placement_relation",
                "Restore the current bound horizontal placement relation, using its native footprint geometry and thresholds; preserve holding and observed transport clearance. Do not target a fixed center or release yet.",
                ("forward", "backward", "left", "right", "hold"), LocalIntentConditionV1("placement_horizontal_relation_satisfied", "equals", True)),
        }
        if mode in recovery:
            name, objective, allowed, completion = recovery[mode]
            actions = tuple(action for action in allowed if action in skill.allowed_actions)
            if actions:
                candidates.append(LocalIntentCandidateV1(name, objective, actions, "runtime_conditions",
                    reference_ref=skill.target_ref, maintain=("subject_held", "avoid_other_objects"),
                    completion_conditions=(LocalIntentConditionV1("subject_held", "equals", True), completion)))
            return tuple(candidates)
        if mode == "unknown":
            return (LocalIntentCandidateV1("observe_placement_evidence",
                "Observe missing or stale current placement/holding evidence; unknown does not authorize translation or release.",
                ("hold",), "phase_contract", reference_ref=skill.target_ref),)
        candidates.extend(placement_preparation_candidates_v1(snapshot,
            step_m=placement_preparation_step_m, tolerance_m=placement_preparation_tolerance_m,
            minimum_improvement_m=placement_preparation_minimum_improvement_m, retained=placement_preparation))
        interval = placement_height_interval_v1(snapshot)
        height = None if interval is None else local_intent_interval_value_v1(interval, snapshot)
        actions = tuple(action for action in skill.allowed_actions if action in ("up", "down", "hold"))
        if mode == "placement" and interval is not None and height is not None and actions:
            if placement_horizontal_preparation_allowed_v1(snapshot):
                actions = tuple(action for action in skill.allowed_actions if action != "gripper_open")
            candidates.append(LocalIntentCandidateV1("lower_to_release",
                "Prepare placement by adjusting signed bottom-to-support clearance into the coarse interval while preserving the horizontal relation. At verified transport clearance, optional horizontal actions can improve the native relation margins before descent. Completion ends this preparation, not release or final placement.",
                actions, "runtime_conditions", reference_ref=skill.target_ref, maintain=phase.maintain,
                completion_conditions=(LocalIntentConditionV1("subject_held", "equals", True),
                    LocalIntentConditionV1("placement_horizontal_relation_satisfied", "equals", True),
                    LocalIntentConditionV1(interval.measurement, "at_least", interval.minimum_m),
                    LocalIntentConditionV1(interval.measurement, "at_most", "placement_height_adjustment_maximum_clearance_m")),
                progress_interval=interval))
        elif mode == "placement" and "hold" in skill.allowed_actions:
            candidates.append(LocalIntentCandidateV1("observe_placement_evidence",
                "Observe missing current support-height evidence. Do not translate using an old height interval; release remains a separate model choice, not a verified safe outcome.",
                ("hold",), "phase_contract", reference_ref=skill.target_ref))
        support_interval = placement_height_interval_v1(snapshot, maximum_measurement="placement_support_approach_maximum_clearance_m")
        support_height = None if support_interval is None else local_intent_interval_value_v1(support_interval, snapshot)
        if mode == "placement" and support_height is not None:
            support_actions = tuple(action for action in skill.allowed_actions if action in ("up", "down", "hold"))
            if support_actions:
                candidates.append(LocalIntentCandidateV1("approach_placement_support",
                    "Optionally approach the observed target support using the named support-height tolerance. This is a local height goal, not physical contact, INSIDE completion or a requirement before opening. Compare drop distance and current relation margins; do not assume a safe tool path.",
                    support_actions, "runtime_conditions", reference_ref=skill.target_ref, maintain=phase.maintain,
                    completion_conditions=(LocalIntentConditionV1("subject_held", "equals", True),
                        LocalIntentConditionV1("placement_horizontal_relation_satisfied", "equals", True),
                        LocalIntentConditionV1(support_interval.measurement, "at_least", 0.0),
                        LocalIntentConditionV1(support_interval.measurement, "at_most", "placement_support_approach_maximum_clearance_m")),
                    progress_interval=support_interval))
        actions = tuple(action for action in skill.allowed_actions if action in ("gripper_open", "hold"))
        if actions:
            candidates.append(LocalIntentCandidateV1("release_subject",
                "Release the subject by opening the gripper when observed placement geometry supports release; the local height interval is not a veto on opening. Runtime verifies the outcome.",
                actions, "phase_contract", reference_ref=skill.target_ref, maintain=phase.maintain))
    else:
        for name in skill.allowed_intents:
            candidates.append(LocalIntentCandidateV1(name, phase.objective, skill.allowed_actions,
                "phase_contract", reference_ref=skill.target_ref or skill.subject_ref, maintain=phase.maintain))
    return tuple(candidates)


class LocalIntentRuntimeV1:
    """No action/side selection occurs here; only the model's valid choice is retained."""

    def __init__(self, config: LocalIntentConfigV1, *, placement_preparation_step_m=None,
                 placement_preparation_tolerance_m=None):
        self.config = config
        values = (placement_preparation_step_m, placement_preparation_tolerance_m)
        if (any(v is None for v in values) != all(v is None for v in values)
            or any(v is not None and (not isfinite(v) or v<=0) for v in values)):
            raise ValueError("placement preparation requires both positive deployment parameters")
        self._placement_preparation_step_m = placement_preparation_step_m
        self._placement_preparation_tolerance_m = placement_preparation_tolerance_m
        self._next_id = 1
        self.reset()

    def reset(self):
        self._key = None
        self._execution = None
        self._retained_approach = None
        self._handoff = None
        self._handoff_reference = None
        self._last_termination = None
        self._phase_closed = False
        self._push_contact_plan = None
        self._push_preparation = None
        self._push_goal_execution = None
        self._selected_observation_id = None
        self._replans = 0
        self._blocked_reason = None

    def context(self, snapshot: StateSnapshotV2) -> LocalIntentContextV1:
        key = self._scope(snapshot)
        # 共享循环会在 close_phase 后再次附加原阶段的终止快照；
        # 此时计划尚待交接，不能按目标阶段匹配失败将其清空。
        if (self._push_contact_plan is not None and not (self._phase_closed and key == self._key)
            and current_push_plan_side_v1(self._push_contact_plan, snapshot) is None):
            self._push_contact_plan = None
        if self._push_preparation is not None and current_push_plan_side_v1(self._push_preparation.plan, snapshot) is None:
            self._push_preparation = None
        if key != self._key:
            if not self._phase_closed and self._execution is not None:
                self._terminate(snapshot, "cancelled", "scope changed without explicit phase completion", "scope_changed")
            if (self._key is None or key[0] != self._key[0] or key[2:] != self._key[2:]
                or self._handoff is None or key[1] != self._handoff.destination_phase):
                self._handoff = None
                self._handoff_reference = None
            self._retained_approach = None  # 本阶段选过的计划不直接跨阶段继承。
            self._push_preparation = None
            self._push_goal_execution = None
            self._key = key
            self._execution = None
            self._replans = 0
            self._blocked_reason = None
            self._phase_closed = False
        if snapshot.grasp_approach is not None and snapshot.grasp_approach.blocked_reason is not None:
            self._blocked_reason = snapshot.grasp_approach.blocked_reason
        if self._phase_closed:
            return LocalIntentContextV1(snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id,
                snapshot.skill.skill_instance_id, snapshot.skill.phase, (), self._execution,
                self._replans, self._blocked_reason, last_termination=self._last_termination,
                push_goal_execution=self._push_goal_execution)
        if (self._push_goal_execution is not None
            and self._push_goal_execution.status in ("active", "suspended")):
            return self._goal_plan_context(snapshot)
        if self._handoff is not None:
            reached = self._reached_approach(self._handoff_reference, snapshot)
            self._handoff = None if reached is None else replace(self._handoff, candidate=reached[0],
                metric_m=reached[1], based_on_state_id=snapshot.state_id,
                based_on_action_epoch=snapshot.action_epoch,
                based_on_observation_id=snapshot.visual_scene.observation_id)
            if self._handoff is None:
                self._handoff_reference = None
        if snapshot.skill.skill_kind == "surface_push" and not observed_contact_sides_v1(snapshot):
            self._retained_approach = None
        inherited = None if self._handoff is None else self._handoff.candidate
        retained_placement = (self._execution.candidate if self._execution is not None
            and self._execution.status=="active" and self._execution.candidate.placement_preparation is not None else None)
        placement_arguments = dict(placement_preparation=retained_placement,
            placement_preparation_step_m=self._placement_preparation_step_m,
            placement_preparation_tolerance_m=self._placement_preparation_tolerance_m,
            placement_preparation_minimum_improvement_m=self.config.minimum_progress_m)
        candidates = local_intent_candidates_v1(snapshot, retained_approach=inherited,
            push_contact_plan=self._push_contact_plan, push_preparation=self._push_preparation, **placement_arguments)
        execution = self._execution
        if execution is not None and execution.status == "active":
            geometry = local_intent_candidates_v1(snapshot, for_validation=True, retained_approach=inherited,
                push_contact_plan=self._push_contact_plan, push_preparation=self._push_preparation, **placement_arguments)
            replacement = next((item for item in geometry if item.choice_id == execution.candidate.choice_id), None)
            if replacement is not None and execution.candidate.metric_kind == "runtime_conditions":
                execution = replace(execution, candidate=replace(execution.candidate,
                    allowed_actions=replacement.allowed_actions))
            if replacement is not None and (snapshot.skill.phase == "push_toward_goal"
                or execution.candidate.choice_id.startswith("prepare_grasp_view_")):
                execution = replace(execution, candidate=replacement)
            metric = local_intent_metric_v1(execution.candidate, snapshot)
            reason, status = "intent remains valid", "active"
            if not snapshot.freshness.visual_is_fresh:
                status, reason = "invalidated", "fresh observation required to continue intent"
            elif execution.candidate.metric_kind == "runtime_conditions" and local_intent_condition_status_v1(execution.candidate, snapshot) is True:
                status, reason = "completed", "current Runtime measurements satisfy the local condition; phase verification remains Runtime-owned"
            elif (execution.candidate.metric_kind in ("phase_contract", "contact_geometry")
                and snapshot.phase_feedback is not None and snapshot.phase_feedback.outcome == "passed"):
                status, reason = "completed", "Runtime phase contract passed"
            elif replacement is None:
                status, reason = "invalidated", "intent geometry/eligibility is no longer observable"
            elif self._reference_shift(execution.candidate, replacement) > self.config.maximum_reference_shift_m:
                status, reason = "invalidated", "observed reference geometry moved beyond intent validity margin"
            elif (execution.candidate.metric_kind in ("tcp_xy", "tcp_z")
                and metric is not None and metric <= execution.candidate.tolerance_m
                and (not execution.candidate.choice_id.startswith("prepare_grasp_view_")
                    or grasp_preparation_goal_reached_v1(execution.candidate, snapshot))
                and (execution.candidate.placement_preparation is None
                    or placement_preparation_goal_reached_v1(execution.candidate,snapshot))):
                status, reason = "completed", "local geometric condition reached; phase verification remains Runtime-owned"
            elif not any(item.choice_id == execution.candidate.choice_id for item in candidates):
                status, reason = "invalidated", "current Runtime action permissions no longer permit this intent"
            count, best = execution.nonprogress_actions, execution.best_metric_m
            epoch = execution.last_observed_action_epoch
            action = snapshot.recent_action
            if (execution.candidate.progress_interval is not None and status == "completed" and metric is not None):
                best, count, epoch = metric if best is None else min(best, metric), 0, snapshot.action_epoch
            if (status == "active" and metric is not None and action is not None
                and snapshot.action_epoch > epoch
                and snapshot.visual_scene.capture_timestamp_s >= action.end_timestamp_s):
                epoch = snapshot.action_epoch
                if best is None or best - metric >= self.config.minimum_progress_m:
                    best, count = metric, 0
                else:
                    count += 1
                if count >= self.config.maximum_actions_without_new_best:
                    status, reason = "invalidated", "repeated actions produced no new best local geometric progress"
            self._execution = replace(execution, status=status, reason=reason, metric_m=metric,
                best_metric_m=best, nonprogress_actions=count, last_observed_action_epoch=epoch)
            if status == "invalidated":
                self._push_preparation = None
                if execution.candidate.choice_id.startswith("approach_edge_"):
                    self._retained_approach = None
                self._replans += 1
                if self._replans >= self.config.maximum_local_replans:
                    self._blocked_reason = "local intent replanning budget exhausted; higher-level replanning required"
        # 仅新鲜核验完成本步骤后推进已选准备义务；没有选择新的侧面或动作。
        preparation = self._push_preparation
        if preparation is not None:
            side = current_push_plan_side_v1(preparation.plan, snapshot)
            if preparation.stage == "clearance" and push_reposition_ready_v1(snapshot) is True:
                preparation = replace(preparation, stage="align_side")
            if (preparation.stage == "align_side" and push_reposition_ready_v1(snapshot) is True
                and side is not None and push_side_aligned_v1(snapshot, side)):
                preparation = replace(preparation, stage="contact")
            if preparation.stage == "contact" and push_interaction_mode_v1(snapshot) == "advance":
                self._push_contact_plan = preparation.plan
                preparation = None
            self._push_preparation = preparation
        if snapshot.skill.phase == "push_toward_goal":
            candidates = local_intent_candidates_v1(snapshot, push_contact_plan=self._push_contact_plan,
                push_preparation=self._push_preparation)
        if self._retained_approach is not None and snapshot.skill.phase == "align_precontact":
            geometry = local_intent_candidates_v1(snapshot, for_validation=True)
            retained = next((item for item in geometry if item.choice_id == self._retained_approach.choice_id), None)
            if (retained is None or self._reference_shift(self._retained_approach, retained)
                > self.config.maximum_reference_shift_m):
                self._retained_approach = None
            else:
                # Retain the side actually chosen by JEV; do not pick a new side on its behalf.
                candidates = tuple(item for item in candidates if not item.choice_id.startswith("approach_edge_")
                    or item.choice_id == self._retained_approach.choice_id)
        pending = tuple(item for item in candidates if item.metric_kind in ("phase_contract", "contact_geometry")
            or (item.metric_kind == "runtime_conditions" and local_intent_condition_status_v1(item, snapshot) is not True)
            or (item.metric_kind in ("tcp_xy", "tcp_z") and local_intent_metric_v1(item, snapshot) is not None
                and (not placement_preparation_goal_reached_v1(item,snapshot) if item.placement_preparation is not None
                    else not grasp_preparation_goal_reached_v1(item, snapshot) if item.choice_id.startswith("prepare_grasp_view_")
                    else local_intent_metric_v1(item, snapshot) > item.tolerance_m)))
        return LocalIntentContextV1(snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id,
            snapshot.skill.skill_instance_id, snapshot.skill.phase, pending, self._execution,
            self._replans, ("current push prerequisite is explicitly false" if push_interaction_mode_v1(snapshot) == "blocked"
                else self._blocked_reason), self._retained_approach,
            self._handoff, self._last_termination, self._push_contact_plan, self._push_preparation,
            self._push_goal_execution)

    def _goal_plan_context(self, snapshot):
        goal = self._push_goal_execution
        stage, candidate, mode = push_plan_step_v1(goal, snapshot)
        if candidate is None or mode == "blocked":
            self.request_replan("persistent plan lost current side correspondence or a prerequisite failed")
            return self.context(snapshot)
        current = native_push_goal_observation_v1(snapshot)
        previous = goal.current_goal
        best, count, epoch = goal.best_goal, goal.actions_without_goal_progress, goal.last_counted_action_epoch
        exit_best = goal.best_exit_progress
        if best is None and current is not None:
            best = current  # 首次可用证据建立基线，不伪装成一次进展。
        action = snapshot.recent_action
        counted_nonprogress = False
        attempted_advance = False
        if (current is not None and previous is not None and action is not None
            and previous.action_epoch == action.start_action_epoch
            and snapshot.action_epoch == action.end_action_epoch
            and snapshot.action_epoch > epoch and self._fresh_after_action(snapshot)):
            epoch = snapshot.action_epoch
            attempted_advance = goal.stage == "advance"
            direction = tuple(-v for v in goal.plan.selected_side.outward_normal_xy)
            improved = native_push_goal_improved_v1(best, current, self.config.minimum_progress_m,
                direction_xy=direction, best_exit_progress=exit_best)
            if improved is True:
                best, count = current, 0
            elif improved is False and attempted_advance:
                count += 1
                counted_nonprogress = True
            exit_best = update_best_exit_progress_v1(exit_best, current, self.config.minimum_progress_m)
        suspended = mode == "unknown"
        goal = replace(goal, stage=stage, status="suspended" if suspended else "active",
            reason="current evidence missing; retain historical plan without motion permission" if suspended
                else "continue the original model-selected goal plan",
            baseline=goal.baseline if goal.baseline is not None else current,
            current_goal=current,
            best_goal=best, actions_without_goal_progress=count,
            last_counted_action_epoch=epoch, advancement_started=goal.advancement_started or attempted_advance,
            best_exit_progress=exit_best)
        self._push_goal_execution = goal
        if counted_nonprogress and count >= self.config.maximum_actions_without_new_best and not suspended:
            self.request_replan("persistent pushing produced no new best native goal progress")
            return self.context(snapshot)
        execution = self._execution
        same_step = execution is not None and execution.status == "active" and execution.candidate.choice_id == candidate.choice_id
        metric = local_intent_metric_v1(candidate, snapshot)
        if not same_step:
            execution = LocalIntentExecutionV1(self._next_id, candidate,
                goal.plan.selected_state_id, goal.plan.selected_action_epoch,
                "active", "derived current obligation of the JEV-selected persistent push plan",
                metric, metric, 0, snapshot.action_epoch, "push_plan_step", goal.plan.selected_state_id)
            self._next_id += 1
        else:
            local_count, local_best, local_epoch = execution.nonprogress_actions, execution.best_metric_m, execution.last_observed_action_epoch
            if metric is not None and action is not None and snapshot.action_epoch > local_epoch and self._fresh_after_action(snapshot):
                local_epoch = snapshot.action_epoch
                if local_best is None or local_best - metric >= self.config.minimum_progress_m:
                    local_best, local_count = metric, 0
                else:
                    local_count += 1
            execution = replace(execution, candidate=candidate, metric_m=metric,
                best_metric_m=local_best, nonprogress_actions=local_count, last_observed_action_epoch=local_epoch)
            self._execution = execution
            if local_count >= self.config.maximum_actions_without_new_best and not suspended:
                self.request_replan("persistent plan step produced no new best local progress")
                return self.context(snapshot)
        self._execution = execution
        self._push_contact_plan = goal.plan
        self._push_preparation = None if stage == "advance" or suspended else PushPreparationV1(goal.plan, stage)
        return LocalIntentContextV1(snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id,
            snapshot.skill.skill_instance_id, snapshot.skill.phase, (candidate,), execution,
            self._replans, self._blocked_reason, push_contact_plan=goal.plan,
            push_preparation=self._push_preparation, push_goal_execution=goal)

    def close_phase(self, snapshot: StateSnapshotV2, next_skill):
        """原 Runtime 已结束阶段；结束意图不会重新裁决该阶段。"""
        feedback = snapshot.phase_feedback
        contract = next(item.completion_contract for item in
            CAPABILITY_TEMPLATES_V1[CapabilityKind(snapshot.skill.skill_kind)].phases
            if item.phase_id == snapshot.skill.phase)
        if (feedback is None or feedback.outcome not in ("passed", "failed")
            or feedback.contract_id != contract
            or (feedback.skill_instance_id, feedback.phase, feedback.based_on_state_id,
                feedback.based_on_action_epoch, feedback.based_on_observation_id)
            != (snapshot.skill.skill_instance_id, snapshot.skill.phase, snapshot.state_id,
                snapshot.action_epoch, snapshot.visual_scene.observation_id)):
            raise ValueError("phase termination requires current terminal Runtime feedback")
        key = self._scope(snapshot)
        if self._key != key:
            self.context(snapshot)
        if self._phase_closed:
            return None  # 重复附加同一反馈不能再生成结束事件。
        self._handoff = None
        self._handoff_reference = None
        if (feedback.outcome == "passed" and snapshot.skill.phase == "establish_contact"
            and next_skill is not None and next_skill.phase == "push_toward_goal"
            and self._execution is not None and self._execution.candidate.contact_side is not None
            and self._selected_observation_id is not None
            and current_runtime_measurements_v1(snapshot).get("supported_push_contact") is True):
            skill = snapshot.skill
            side = next((side for side in observed_contact_sides_v1(snapshot)
                if side.edge_index == self._execution.candidate.contact_side.edge_index), None)
            if side is not None:
                plan = PushContactPlanV1(skill.skill_instance_id, skill.subject_ref, skill.source_ref,
                    skill.target_ref, None if skill.goal is None else (skill.goal.predicate.value,
                        skill.goal.reference_ref, skill.goal.desired_value), self._execution.candidate.contact_side,
                    self._execution.selected_state_id, self._execution.selected_action_epoch, self._selected_observation_id)
                self._push_contact_plan = plan if plan.matches(next_skill) else None
        if (feedback.outcome == "passed" and snapshot.skill.phase == "align_precontact"
            and next_skill is not None and next_skill.phase == "establish_contact"
            and next_skill.skill_instance_id == snapshot.skill.skill_instance_id
            and next_skill.goal == snapshot.skill.goal and next_skill.subject_ref == snapshot.skill.subject_ref
            and next_skill.source_ref == snapshot.skill.source_ref and next_skill.target_ref == snapshot.skill.target_ref
            and next_skill.skill_kind == snapshot.skill.skill_kind
            and self._retained_approach is not None):
            reached = self._reached_approach(self._retained_approach, snapshot)
            if reached is not None:
                self._handoff = LocalIntentHandoffV1(reached[0], snapshot.skill.skill_instance_id,
                    snapshot.skill.phase, next_skill.phase, snapshot.state_id, snapshot.action_epoch,
                    snapshot.visual_scene.observation_id, reached[1])
                self._handoff_reference = self._retained_approach
        termination = None
        if self._execution is not None:
            candidate = self._execution.candidate
            best = self._execution.best_metric_m
            status, reason = "cancelled", "Runtime phase failed; local intent ended"
            metric = local_intent_metric_v1(candidate, snapshot)
            if feedback.outcome == "passed":
                status, reason = "superseded", "Runtime phase passed without proving the selected local point reached"
                if candidate.metric_kind in ("phase_contract", "contact_geometry"):
                    status, reason = "completed", "corresponding actual Runtime phase contract passed"
                elif candidate.metric_kind == "runtime_conditions":
                    if local_intent_condition_status_v1(candidate, snapshot) is True:
                        status, reason = "completed", "current Runtime measurements satisfy local completion at phase end"
                elif self._fresh_after_action(snapshot):
                    replacement = next((item for item in local_intent_candidates_v1(snapshot, for_validation=True,
                        placement_preparation=candidate if candidate.placement_preparation is not None else None)
                        if item.choice_id == candidate.choice_id), None)
                    if replacement is not None and self._reference_shift(candidate, replacement) <= self.config.maximum_reference_shift_m:
                        metric = local_intent_metric_v1(replacement, snapshot)
                        if metric is not None:
                            best = metric if best is None else min(best, metric)
                        if (metric is not None and metric <= candidate.tolerance_m
                            and (candidate.placement_preparation is None
                                or placement_preparation_goal_reached_v1(candidate,snapshot))):
                            status, reason = "completed", "local geometric condition reached in phase-end observation"
            self._execution = replace(self._execution, metric_m=metric, best_metric_m=best)
            termination = self._terminate(snapshot, status, reason, "phase_" + feedback.outcome)
        self._retained_approach = None
        self._phase_closed = True
        if self._push_goal_execution is not None:
            self._push_goal_execution = replace(self._push_goal_execution,
                status="completed" if feedback.outcome == "passed" else "invalidated",
                reason="original Runtime phase contract passed" if feedback.outcome == "passed" else "Runtime phase failed")
        return termination

    def _terminate(self, snapshot, status, reason, trigger):
        self._execution = replace(self._execution, status=status, reason=reason,
            last_observed_action_epoch=snapshot.action_epoch)
        self._last_termination = LocalIntentTerminationV1(self._execution, self._key[0], self._key[1],
            snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id, trigger)
        return self._last_termination

    def stop(self, snapshot: StateSnapshotV2, reason: str):
        """回合最终停止时结束命令，不伪造阶段失败或改变恢复路径。"""
        if self._execution is None or self._phase_closed:
            return None
        status = "cancelled" if self._execution.status == "active" else self._execution.status
        self._execution = replace(self._execution, metric_m=(
            local_intent_metric_v1(self._execution.candidate, snapshot)
            if self._fresh_after_action(snapshot) else None))
        termination = self._terminate(snapshot, status, reason, "run_stopped")
        self._retained_approach = None
        self._handoff = None
        self._handoff_reference = None
        self._phase_closed = True
        self._push_contact_plan = None
        self._push_preparation = None
        if self._push_goal_execution is not None:
            self._push_goal_execution = replace(self._push_goal_execution, status="cancelled", reason=reason)
        return termination

    @staticmethod
    def _scope(snapshot):
        skill = snapshot.skill
        return (skill.skill_instance_id, skill.phase, skill.goal, skill.subject_ref,
            skill.source_ref, skill.target_ref, skill.skill_kind)

    @staticmethod
    def _fresh_after_action(snapshot):
        return (snapshot.freshness.visual_is_fresh and (snapshot.recent_action is None
            or snapshot.visual_scene.capture_timestamp_s >= snapshot.recent_action.end_timestamp_s))

    def _reached_approach(self, candidate, snapshot):
        if (not self._fresh_after_action(snapshot) or candidate.metric_kind != "tcp_xy"
            or candidate.contact_side is None or candidate.reference_ref != snapshot.skill.subject_ref):
            return None
        side = next((item for item in observed_contact_sides_v1(snapshot)
            if item.edge_index == candidate.contact_side.edge_index), None)
        if side is None:
            return None
        original = candidate.contact_side
        # 原选择的外侧距离不变；用本次观测的侧面重建 XY，不能拿旧点证明已达。
        clearance = sum((candidate.target_robot_base_m[i] -
            (original.start_xy_m[i] + original.end_xy_m[i]) * 0.5) * original.outward_normal_xy[i]
            for i in (0, 1))
        point = tuple((side.start_xy_m[i] + side.end_xy_m[i]) * 0.5
            + side.outward_normal_xy[i] * clearance for i in (0, 1)) + (snapshot.robot.tcp_position_robot_base_m[2],)
        current = replace(candidate, contact_side=side, target_robot_base_m=point)
        if self._reference_shift(candidate, current) > self.config.maximum_reference_shift_m:
            return None
        metric = local_intent_metric_v1(current, snapshot)
        return (current, metric) if metric <= candidate.tolerance_m else None

    @staticmethod
    def _reference_shift(first, second):
        if first.reference_ref != second.reference_ref:
            return float("inf")
        if first.contact_side is not None and second.contact_side is not None:
            if sum(a * b for a, b in zip(first.contact_side.outward_normal_xy,
                second.contact_side.outward_normal_xy)) < 1 - 1e-6:
                return float("inf")
            return max(hypot(*(a - b for a, b in zip(old, new)))
                for old, new in ((first.contact_side.start_xy_m, second.contact_side.start_xy_m),
                    (first.contact_side.end_xy_m, second.contact_side.end_xy_m)))
        if first.target_robot_base_m is None or second.target_robot_base_m is None:
            return 0.0
        axes = (0, 1) if first.metric_kind == "tcp_xy" else (2,)
        return sum((first.target_robot_base_m[i] - second.target_robot_base_m[i]) ** 2 for i in axes) ** 0.5

    def accept(self, choice: LocalIntentChoiceV1, snapshot: StateSnapshotV2):
        context = snapshot.local_intent
        if (context is None or context.blocked_reason is not None
            or choice.based_on_state_id != snapshot.state_id
            or choice.based_on_action_epoch != snapshot.action_epoch
            or choice.skill_instance_id != snapshot.skill.skill_instance_id
            or choice.phase != snapshot.skill.phase):
            raise ValueError("stale or invalid local intent choice")
        candidate = next((item for item in context.candidates if item.choice_id == choice.choice_id), None)
        if candidate is None:
            raise ValueError("model selected an unknown local intent candidate")
        if context.execution is not None and context.execution.status == "active":
            if context.execution.candidate.choice_id != choice.choice_id:
                raise ValueError("cannot silently replace an active local intent")
            return context.execution
        metric = local_intent_metric_v1(candidate, snapshot)
        self._execution = LocalIntentExecutionV1(self._next_id, candidate, snapshot.state_id,
            snapshot.action_epoch, "active", "validated JEV-selected local intent", metric, metric, 0, snapshot.action_epoch)
        self._next_id += 1
        self._selected_observation_id = snapshot.visual_scene.observation_id
        preparation_stages = {"prepare_push_side_edge_": "clearance", "align_reprepared_push_edge_": "align_side",
            "contact_reprepared_push_edge_": "contact"}
        preparation_stage = next((stage for prefix, stage in preparation_stages.items() if candidate.choice_id.startswith(prefix)), None)
        if (candidate.choice_id.startswith("restore_push_contact_edge_") or preparation_stage is not None) and candidate.contact_side is not None:
            skill = snapshot.skill
            plan = PushContactPlanV1(skill.skill_instance_id, skill.subject_ref,
                skill.source_ref, skill.target_ref, None if skill.goal is None else
                (skill.goal.predicate.value, skill.goal.reference_ref, skill.goal.desired_value),
                candidate.contact_side, snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id)
            if preparation_stage is not None:
                # 后续步骤沿用最初的JEV选择身份，不刷新为伪造的新选择。
                if self._push_preparation is None:
                    self._push_preparation = PushPreparationV1(plan, preparation_stage)
            else:
                self._push_contact_plan = plan
        if snapshot.skill.phase == "push_toward_goal" and candidate.contact_side is not None:
            skill = snapshot.skill
            plan = PushContactPlanV1(skill.skill_instance_id, skill.subject_ref, skill.source_ref,
                skill.target_ref, None if skill.goal is None else (skill.goal.predicate.value,
                    skill.goal.reference_ref, skill.goal.desired_value), candidate.contact_side,
                snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id)
            previous = self._push_goal_execution
            current = native_push_goal_observation_v1(snapshot)
            self._push_goal_execution = PushGoalExecutionV1(plan, candidate.choice_id,
                preparation_stage or ("advance" if candidate.choice_id == "advance_surface_push" else "contact"),
                "active", "validated JEV selection of a goal-bound push plan",
                baseline=current if previous is None else previous.baseline,
                current_goal=current, best_goal=current if previous is None else previous.best_goal,
                actions_without_goal_progress=0 if previous is None else previous.actions_without_goal_progress,
                last_counted_action_epoch=snapshot.action_epoch,
                advancement_started=previous is not None and previous.advancement_started,
                best_exit_progress=(() if current is None else current.exit_progress)
                    if previous is None else previous.best_exit_progress)
        if candidate.choice_id.startswith("approach_edge_"):
            self._retained_approach = candidate
        return self._execution

    def request_replan(self, reason: str):
        if self._execution is None or self._execution.status != "active":
            raise ValueError("local replanning requires an active intent")
        self._execution = replace(self._execution, status="invalidated", reason=reason)
        self._push_preparation = None
        if self._push_goal_execution is not None:
            self._push_goal_execution = replace(self._push_goal_execution, status="invalidated", reason=reason)
        self._replans += 1
        if self._execution.candidate.choice_id.startswith("approach_edge_"):
            self._retained_approach = None
        if self._replans >= self.config.maximum_local_replans:
            self._blocked_reason = "local intent replanning budget exhausted; higher-level replanning required"
