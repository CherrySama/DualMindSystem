from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any
from jev4mujoco.contracts.actions import AtomicAction, TRANSLATION_DIRECTIONS_ROBOT_BASE

from jev4mujoco.runtime.progress import pick_xy_error_v1, placement_horizontal_measurements_v1
from jev4mujoco.runtime.grasp_geometry import grasp_pose_correction_v1, grasp_translation_step_v1
from jev4mujoco.runtime.grasp_observation import grasp_approach_context_v1
from jev4mujoco.runtime.placement_preparation import placement_preparation_context_v1
from jev4mujoco.runtime.local_intent import (local_intent_correction_v1, local_intent_metric_v1,
    local_intent_conditions_v1, placement_relation_restore_mode_v1, current_runtime_measurements_v1,
    apply_placement_action_permissions_v1, placement_height_interval_v1,
    tool_retraction_motion_ready_v1,
    task_observation_motion_ready_v1,
    local_intent_interval_value_v1, local_intent_interval_distance_v1)
from jev4mujoco.runtime.tool_geometry import (contact_geometry_v1, observed_contact_sides_v1,
    tool_transfer_clearance_v1)
from jev4mujoco.runtime.push_interaction import (
    push_interaction_mode_v1, current_push_plan_side_v1, push_goal_geometry_v1)
from jev4mujoco.runtime.observation_geometry import (
    required_task_observation_refs_v1, current_visibility_diagnostic_v1)
from jev4mujoco.contracts.state_snapshot import GraspState, StateSnapshotV2
from jev4mujoco.planning.phase_definitions import (
    bound_subtask_definition_v1, GOAL_DEFINITIONS_V1, PHASE_MEASUREMENT_DEFINITIONS_V1,
)
from jev4mujoco.contracts.visual_scene import (
    EntityV1,
    EvidenceValueV1,
    RegionV1,
    VisualPredicate,
    Visibility,
)


@dataclass(frozen=True, slots=True)
class JevDecisionContextV1:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _evidence(value: EvidenceValueV1) -> dict[str, Any]:
    return {
        "value": _json_value(value.value),
        "status": value.status.value,
        "confidence": value.confidence,
    }


def _entity(entity: EntityV1) -> dict[str, Any]:
    return {
        "ref": entity.track_id,
        "type": entity.entity_type.value,
        "category": _evidence(entity.semantic.category),
        "color": _evidence(entity.semantic.color),
        "visibility": entity.visibility.value,
        "centroid_robot_base_m": _evidence(
            entity.geometry.centroid_robot_base_m
        ),
        "top_center_robot_base_m": _evidence(
            entity.geometry.top_center_robot_base_m
        ),
        "extent_m": _evidence(entity.geometry.extent_m),
        "footprint_xy_robot_base_m": _evidence(entity.geometry.footprint_xy_robot_base_m),
        "surface_normal_robot_base": _evidence(entity.geometry.surface_normal_robot_base),
        "support_region_ref": entity.support_region_ref,
        "measurement": asdict(entity.measurement),
    }


def _optional_evidence(value: EvidenceValueV1 | None) -> dict[str, Any] | None:
    return None if value is None else _evidence(value)


def _observation_basis_payload(basis, *, current=False):
    if basis is None:
        return None
    result = asdict(basis)
    # 许可不是测量作用域；同阶段允许动作的合法变化不抹去旧观测。
    result["scope"] = {name: getattr(basis.skill, name) for name in (
        "skill_instance_id", "skill_kind", "phase", "subject_ref", "source_ref", "target_ref")}
    result["scope"]["goal"] = None if basis.skill.goal is None else asdict(basis.skill.goal)
    result.pop("skill")
    result["measurement_evidence_current"] = dict(basis.measurement_evidence_current)
    if current:
        # 当前身份/时间复用外层 state、observation 和 phase.evidence_basis；历史则独立携带。
        for name in ("state_id", "action_epoch", "observation_id", "timestamp_s",
                     "sensor_id", "calibration_id", "evidence_basis"):
            result.pop(name)
    return result


def _region(region: RegionV1) -> dict[str, Any]:
    geometry = region.geometry
    return {
        "ref": region.region_id,
        "type": region.region_type.value,
        "semantic_label": region.semantic_label,
        "owner_entity_ref": region.owner_entity_ref,
        "source": {"kind": region.source.kind.value, "confidence": region.source.confidence},
        "surface_normal_robot_base": _optional_evidence(geometry.surface_normal_robot_base),
        "boundary_xy_robot_base_m": _optional_evidence(
            geometry.boundary_xy_robot_base_m
        ),
        "surface_height_robot_base_m": _optional_evidence(
            geometry.surface_height_robot_base_m
        ),
        "minimum_z_robot_base_m": _optional_evidence(
            geometry.minimum_z_robot_base_m
        ),
        "maximum_z_robot_base_m": _optional_evidence(
            geometry.maximum_z_robot_base_m
        ),
        "point_robot_base_m": _optional_evidence(
            geometry.point_robot_base_m
        ),
    }


def _subtask_definition(snapshot: StateSnapshotV2):
    skill = snapshot.skill
    goal = skill.goal
    return bound_subtask_definition_v1(
        skill.skill_instance_id, skill.skill_kind, skill.subject_ref, skill.source_ref,
        skill.target_ref, None if goal is None else
        (goal.predicate.value, goal.reference_ref, goal.desired_value),
    )


def _definition_payload(definition) -> dict[str, Any]:
    return {
        "definition_id": definition.instance_id,
        "kind": definition.kind,
        "subject_ref": definition.subject_ref,
        "source_ref": definition.source_ref,
        "target_ref": definition.target_ref,
        "final_goal": None if definition.goal is None else {
            "predicate": definition.goal[0], "reference_ref": definition.goal[1],
            "desired_value": definition.goal[2],
            **asdict(GOAL_DEFINITIONS_V1[definition.goal[0]])},
        "phases": [{"phase_id": phase_id, "completion_contract": contract}
                   for phase_id, contract, _ in definition.phases],
    }


def _dependencies(snapshot, subject, target) -> dict[str, Any]:
    """Close public support/owner dependencies without including raw scene data."""
    refs = {snapshot.skill.source_ref}
    if subject is not None:
        refs.add(subject.support_region_ref)
    if isinstance(target, RegionV1):
        refs.add(target.owner_entity_ref)
    elif isinstance(target, EntityV1):
        refs.add(target.support_region_ref)
    entities = []
    regions = []
    included = {item for item in (snapshot.skill.subject_ref, snapshot.skill.target_ref) if item}
    while refs - included - {None}:
        ref = sorted(refs - included - {None})[0]
        included.add(ref)
        entity = next((item for item in snapshot.visual_scene.entities if item.track_id == ref), None)
        region = next((item for item in snapshot.visual_scene.regions if item.region_id == ref), None)
        if entity is not None:
            entities.append(_entity(entity))
            refs.add(entity.support_region_ref)
        if region is not None:
            regions.append(_region(region))
            refs.add(region.owner_entity_ref)
    return {
        "entities": entities, "regions": regions,
        "other_objects": [_entity(item) for item in snapshot.visual_scene.entities
                          if item.track_id not in included],
    }


def _motion_error(snapshot, subject, target):
    """Only well-defined phase corrections; relation goals need full geometry."""
    phase = snapshot.skill.phase
    kind = snapshot.skill.skill_kind
    role = None
    delta = None
    tolerance = None
    measurements = {} if snapshot.phase_feedback is None else {
        item.name: item.value for item in snapshot.phase_feedback.measurements}
    parameters = {} if snapshot.phase_feedback is None else dict(snapshot.phase_feedback.parameters)
    if kind == "pick" and phase in ("align_subject_xy", "descend_to_grasp"):
        error = pick_xy_error_v1(snapshot, snapshot.skill.subject_ref)
        role = "tcp_to_subject_top_xy"
        delta = None if error is None else [error.delta_x_m, error.delta_y_m, 0.0]
        if phase == "descend_to_grasp":
            correction = grasp_pose_correction_v1(snapshot, snapshot.skill.subject_ref)
            delta = None if correction is None else list(correction)
            role = "tcp_to_subject_grasp_pose"
        tolerance = parameters.get("grasp_xy_tolerance_m")
    elif kind == "press" and phase == "align_press_point":
        role = "tcp_to_press_point_xy"
        if isinstance(target, RegionV1) and target.geometry.point_robot_base_m is not None:
            point = target.geometry.point_robot_base_m.value
            tcp = snapshot.robot.tcp_position_robot_base_m
            delta = None if point is None else [point[0] - tcp[0], point[1] - tcp[1], 0.0]
        tolerance = parameters.get("press_point_xy_tolerance_m")
    elif kind == "transport" and phase == "lift_to_clearance":
        role = "subject_bottom_to_required_clearance"
        deficit = measurements.get("subject_bottom_clearance_deficit_m")
        delta = None if deficit is None else [0.0, 0.0, deficit]
    elif kind == "place" and phase == "retract_after_release":
        role = "finger_tool_to_transfer_clearance"
        deficit = current_runtime_measurements_v1(snapshot).get("tool_transfer_clearance_deficit_m")
        delta = None if deficit is None or not tool_retraction_motion_ready_v1(snapshot) else [0.0, 0.0, deficit]
    elif kind == "place" and phase == "observe_task_geometry":
        role = "task_geometry_observability"
        # 可见跨度缺口不等于沿Z移动的距离，不能输出一个伪造的TCP目标。
        delta = None
    return role, delta, tolerance


def _referenced_target(snapshot: StateSnapshotV2) -> EntityV1 | RegionV1 | None:
    reference = snapshot.skill.target_ref
    if reference is None and snapshot.skill.goal is not None:
        reference = snapshot.skill.goal.reference_ref
    if reference is None:
        return None
    for entity in snapshot.visual_scene.entities:
        if entity.track_id == reference:
            return entity
    for region in snapshot.visual_scene.regions:
        if region.region_id == reference:
            return region
    return None


def _contact_prediction_summary(geometry):
    """保留逐部件预测诊断；侧面定义与当前最近点由当前完整几何提供。"""
    summary = {
        "observable": geometry["observable"],
        "parts": [{name: part[name] for name in (
            "part_id", "normal_gap_m", "tangent_deficit_m", "height_deficit_m",
            "finite_patch_overlap", "local_error_m", "approach_side_status",
        )} for part in geometry["parts"]],
    }
    if "reason" in geometry:
        summary["reason"] = geometry["reason"]
    return summary


def _contact_context(snapshot, side, actions, step_m):
    geometry = contact_geometry_v1(snapshot, side)
    current = geometry["local_error_m"]
    evaluations = []
    values = current_runtime_measurements_v1(snapshot)
    height_delta, height_tolerance = values.get("tcp_to_push_height_z_m"), values.get("push_height_tolerance_m")
    for action in actions:
        direction = ((0.0, 0.0, 0.0) if action == "hold" else
            TRANSLATION_DIRECTIONS_ROBOT_BASE.get(AtomicAction(action)))
        if direction is None:
            continue
        delta = tuple(step_m * value for value in direction)
        predicted = contact_geometry_v1(snapshot, side, delta)
        projected = predicted["local_error_m"]
        height_deficit = (None if height_delta is None or height_tolerance is None else
            max(0.0, abs(height_delta - delta[2]) - height_tolerance))
        evaluations.append({"action": action, "expected_tcp_delta_robot_base_m": delta,
            "projected_original_height_deficit_m": height_deficit,
            "inward_normal_command_m": float(-sum(delta[i] * side.outward_normal_xy[i] for i in (0, 1))),
            "tangential_command_m": float(delta[0] * -side.outward_normal_xy[1] + delta[1] * side.outward_normal_xy[0]),
            "projected_local_error_m": projected,
            "projected_progress_m": None if current is None or projected is None else current - projected,
            "projected_contact_summary": _contact_prediction_summary(predicted), "source": "bounded_command_geometry",
            "is_observed": False, "assumptions": "fixed subject and tool orientation; ignores dynamics and path collisions"})
    return {"geometry": geometry, "action_evaluations": evaluations}


def _native_push_goal_change(before, after):
    """同一关系自身的反事实变化量；不排序、不合成评分、不预测物理跟随。"""
    if before.get("status") != "observed" or after.get("status") not in ("observed", "counterfactual"):
        return None
    if "vertex_outside_distances_m" in before and "vertex_outside_distances_m" in after:
        return {"maximum_outside_distance_reduction_m":
            max(before["vertex_outside_distances_m"]) - max(after["vertex_outside_distances_m"]),
            "sign_meaning": "positive reduces outside distance; negative increases it"}
    if "footprint_clearance_m" in before and "footprint_clearance_m" in after:
        return {"clearance_increase_m": after["footprint_clearance_m"] - before["footprint_clearance_m"],
            "minimum_signed_vertex_distance_increase_m": min(after["vertex_signed_boundary_distances_m"])
                - min(before["vertex_signed_boundary_distances_m"]),
            "overlap_before_after": (before["footprints_overlap"], after["footprints_overlap"]),
            "directional_exit_remaining_reductions_m": tuple({"direction_xy": item["direction_xy"],
                "reduction_m": old["remaining_translation_m"] - item["remaining_translation_m"]}
                for item in after.get("directional_exit_requirements", ())
                for old in before.get("directional_exit_requirements", ())
                if item["direction_xy"] == old["direction_xy"]),
            "sign_meaning": "positive clearance/vertex increase enlarges that distance; positive directional reduction decreases the remaining exit requirement. Clearance stays zero while overlapping; vertex distance alone does not certify clear_of_region"}
    return None


def _push_goal_counterfactual(snapshot, delta):
    geometry = push_goal_geometry_v1(snapshot, delta_xy=delta)
    if geometry["status"] != "observed":
        return geometry
    result = {k: v for k, v in geometry.items() if k not in ("exit_region_ref", "exit_region_boundary_xy_m")}
    result["status"] = "counterfactual"
    if "directional_exit_requirements" in geometry:
        # 完整终点只在当前各侧面计划中出现一次，原子小步保留未排序的变化量。
        result["directional_exit_requirements"] = tuple({k: item[k] for k in (
            "edge_index", "direction_xy", "remaining_translation_m")}
            for item in geometry["directional_exit_requirements"])
    return result


def _push_goal_context(snapshot, actions, step_m):
    current = push_goal_geometry_v1(snapshot)
    observed = None
    feedback = snapshot.phase_feedback
    basis = None if feedback is None else feedback.previous_observation_basis
    action = snapshot.recent_action
    if (current["status"] == "observed" and basis is not None and basis.same_scope_as(snapshot.skill)
        and action is not None and basis.action_epoch == action.start_action_epoch
        and snapshot.action_epoch == action.end_action_epoch):
        old = {m.name: m.value for m in feedback.previous_measurements
            if dict(basis.measurement_evidence_current).get(m.name) is True}
        previous = {name: old["push_" + name] for name in current if "push_" + name in old}
        if previous:
            observed = {"previous_geometry": previous, "source": "fresh_before_after_rgbd_geometry"}
            if "vertex_outside_distances_m" in previous:
                observed["maximum_outside_distance_reduction_m"] = max(previous["vertex_outside_distances_m"]) - max(current["vertex_outside_distances_m"])
            if "footprint_clearance_m" in previous:
                observed["clearance_increase_m"] = current["footprint_clearance_m"] - previous["footprint_clearance_m"]
                observed["overlap_before_after"] = (previous.get("footprints_overlap"), current.get("footprints_overlap"))
    evaluations = []
    if current["status"] == "observed":
        for action in actions:
            direction = ((0.0, 0.0, 0.0) if action == "hold" else
                TRANSLATION_DIRECTIONS_ROBOT_BASE.get(AtomicAction(action)))
            if direction is None:
                continue
            delta = tuple(step_m * value for value in direction)
            counterfactual = _push_goal_counterfactual(snapshot, delta[:2])
            evaluations.append({"action": action, "assumed_subject_delta_xy_m": delta[:2],
                "translated_footprint_counterfactual": counterfactual,
                "native_goal_change": _native_push_goal_change(current, counterfactual),
                "source": "public_footprint_translation_counterfactual", "is_observed": False,
                "is_physical_push_prediction": False})
    goal_execution = None if snapshot.local_intent is None else snapshot.local_intent.push_goal_execution
    selected_exit = None
    if goal_execution is not None:
        direction = tuple(-v for v in goal_execution.plan.selected_side.outward_normal_xy)
        plane = next((p for p in (goal_execution.current_goal.exit_progress
            if goal_execution.current_goal is not None else ()) if p.direction_xy == direction), None)
        best = None if plane is None else next((p for p in goal_execution.best_exit_progress if p.same_plane(plane)), None)
        selected_exit = {"status": "observed" if plane is not None and best is not None else "unknown",
            "current_observation": None if plane is None else asdict(plane),
            "best_observation": None if best is None else asdict(best),
            "actions_without_goal_progress": goal_execution.actions_without_goal_progress,
            "note": "Best is historical evidence for this same direction/region, not current contact or permission. It persists across explicit plan replacements."}
    return {"current": current, "action_evaluations": evaluations, "selected_exit_progress": selected_exit,
        "observed_progress": observed, "continuous_progress_policy_status": (
            "native_goal_plan_progress" if snapshot.local_intent is not None
            and snapshot.local_intent.push_goal_execution is not None else "no_selected_goal_plan"),
        "physical_push_prediction_status": "unsupported",
        "note": "Counterfactual assumes the subject footprint translates by the stated XY displacement, not that the object follows TCP. Separate from fixed-subject tool predictions. No force/friction/slip or selected-face contact verification is available; actual outcome uses next fresh observation and original Runtime conditions."}


def _transport_progress_context(snapshot, actions, step_m):
    feedback = snapshot.phase_feedback
    basis = None if feedback is None else feedback.evidence_basis
    current = placement_horizontal_measurements_v1(snapshot,
        {} if feedback is None else dict(feedback.parameters),
        evidence_current=basis is not None and basis.visual_evidence_current)
    observable = bool(current) and all(item.value is not None for item in current)
    holding_current = feedback is not None and any(condition.measurement == "subject_held"
        and condition.observed is True and condition.evidence_current for condition in feedback.conditions)
    if placement_relation_restore_mode_v1(snapshot) in ("restore_relation", "placement"):
        holding_current = current_runtime_measurements_v1(snapshot).get("subject_held") is True
    prediction_available = (observable and holding_current and basis.robot_evidence_current
        and snapshot.physical.grasp_state is GraspState.HELD
        and snapshot.physical.held_object_ref == snapshot.skill.subject_ref)
    evaluations = []
    if prediction_available:
        for action in actions:
            direction = ((0.0, 0.0, 0.0) if action == "hold" else
                TRANSLATION_DIRECTIONS_ROBOT_BASE.get(AtomicAction(action)))
            if direction is None or direction[2] != 0.0:
                continue
            delta = tuple(step_m * value for value in direction)
            projected = placement_horizontal_measurements_v1(snapshot, dict(feedback.parameters),
                evidence_current=True, delta_xy=delta[:2], predicted=True)
            reductions = []
            margin_changes = []
            approach = {}
            for before, after in zip(current, projected):
                if before.name in ("placement_containment_margin_m", "placement_overlap_fraction_margin"):
                    margin_changes.append({"measurement": before.name, "value": after.value - before.value, "unit": before.unit})
                if before.name == "placement_support_approach_clearance_m":
                    approach["projected_support_approach_clearance_reduction_m"] = before.value - after.value
                if "deficit" not in before.name and before.name != "placement_vertex_outside_distances_m":
                    continue
                reduction = (tuple(a - b for a, b in zip(before.value, after.value))
                    if isinstance(before.value, tuple) else before.value - after.value)
                reductions.append({"measurement": before.name, "value": reduction, "unit": before.unit})
            evaluations.append({"action": action, "expected_tcp_delta_robot_base_m": delta,
                "projected_relation_measurements": [asdict(item) for item in projected],
                "projected_deficit_reductions": reductions,
                "projected_margin_changes": margin_changes,
                **approach,
                "source": "bounded_command_geometry", "is_observed": False,
                "assumptions": "subject follows TCP rigidly; fixed subject orientation and reference; ignores slip, dynamics and path collisions"})
    return {"status": "observed" if observable else "unknown" if current else "unsupported",
        "measurements": [asdict(item) for item in current],
        "prediction_available": bool(prediction_available), "action_evaluations": evaluations}


def _tool_transfer_progress_context(snapshot, actions, step_m):
    values = current_runtime_measurements_v1(snapshot)
    clearance, required = values.get("tool_transfer_clearance_m"), values.get("tool_transfer_required_clearance_m")
    ready = tool_retraction_motion_ready_v1(snapshot)
    error = values.get("tool_transfer_clearance_deficit_m") if ready else None
    evaluations = []
    if ready:
        for action in actions:
            if action not in ("up", "hold"):
                continue
            dz = step_m if action == "up" else 0.0
            after = max(0.0, required - clearance - dz)
            evaluations.append({"action": action, "expected_tcp_delta_robot_base_m": (0.0, 0.0, dz),
                "projected_tool_transfer_clearance_m": clearance + dz,
                "projected_local_error_m": after, "projected_progress_m": error - after,
                "source": "bounded_command_geometry", "is_observed": False,
                "assumptions": "fixed current finger opening and tool orientation; fixed public reference tops; ignores dynamics and path collisions"})
    previous_error = None
    geometry = tool_transfer_clearance_v1(snapshot) if clearance is not None else None
    feedback = snapshot.phase_feedback
    basis = None if feedback is None else feedback.previous_observation_basis
    action = snapshot.recent_action
    if ready and basis is not None and basis.same_scope_as(snapshot.skill) and action is not None:
        validity = dict(basis.measurement_evidence_current)
        old = {m.name: m.value for m in feedback.previous_measurements}
        names = ("grasp_not_held", "gripper_open_observed", "subject_supported", "capability_goal_satisfied")
        if (basis.action_epoch == action.start_action_epoch and snapshot.action_epoch == action.end_action_epoch
            and all(validity.get(name) and old.get(name) is True for name in names)
            and validity.get("tool_transfer_clearance_deficit_m")):
            previous_error = old.get("tool_transfer_clearance_deficit_m")
    return {"status": "observed" if clearance is not None else "unknown",
        "part_minimum_z_m": None if geometry is None else
            [{"part_id": ref, "minimum_z_m": z} for ref, z in geometry.part_minimum_z_m],
        "reference_top_z_m": None if geometry is None else
            [{"reference_ref": ref, "top_z_m": z} for ref, z in geometry.reference_top_z_m],
        "clearance_m": clearance, "required_clearance_m": required, "error_m": error,
        "motion_evidence_ready": ready, "previous_error_m": previous_error,
        "observed_progress_m": None if previous_error is None else previous_error - error,
        "prediction_available": ready, "action_evaluations": evaluations,
        "note": "Finger-tool transfer clearance is separate from release and final stability. It does not certify full-arm avoidance, contact separation or the entire path."}


def _task_observation_context(snapshot, actions, step_m):
    values = current_runtime_measurements_v1(snapshot)
    error = values.get("task_observation_span_deficit_m")
    refs = required_task_observation_refs_v1(snapshot)
    entities = {e.track_id: e for e in snapshot.visual_scene.entities}
    observations = []
    for ref in refs:
        entity = entities.get(ref)
        diagnostic = None if entity is None else current_visibility_diagnostic_v1(entity, snapshot.visual_scene)
        observations.append({"entity_ref": ref, "visibility": None if entity is None else entity.visibility.value,
            "current_fragment_diagnostic": None if diagnostic is None else asdict(diagnostic)})
    previous_error = None
    feedback = snapshot.phase_feedback
    basis = None if feedback is None else feedback.previous_observation_basis
    action = snapshot.recent_action
    if error is not None and basis is not None and basis.same_scope_as(snapshot.skill) and action is not None:
        validity = dict(basis.measurement_evidence_current)
        old = {m.name: m.value for m in feedback.previous_measurements}
        clearance, required = old.get("tool_transfer_clearance_m"), old.get("tool_transfer_required_clearance_m")
        names = ("grasp_not_held", "gripper_open_observed", "subject_supported", "capability_goal_satisfied")
        if (basis.action_epoch == action.start_action_epoch and snapshot.action_epoch == action.end_action_epoch
            and all(validity.get(name) and old.get(name) is True for name in names)
            and validity.get("tool_transfer_clearance_m") and validity.get("tool_transfer_required_clearance_m")
            and clearance is not None and required is not None and clearance >= required
            and validity.get("task_observation_span_deficit_m")):
            previous_error = old.get("task_observation_span_deficit_m")
    return {"required_entity_refs": refs, "current_observations": observations,
        "geometry_observed": values.get("required_task_geometry_observed"), "error_m": error,
        "robot_occlusion_hint": values.get("task_observation_robot_occlusion_hint"),
        "motion_evidence_ready": task_observation_motion_ready_v1(snapshot),
        "previous_error_m": previous_error,
        "observed_progress_m": None if previous_error is None else previous_error - error,
        "prediction_available": False,
        "action_evaluations": [{"action": action, "expected_tcp_delta_robot_base_m": (0.0, 0.0, step_m if action == "up" else 0.0),
            "projected_local_error_m": None, "projected_progress_m": None,
            "source": "bounded_command_geometry", "is_observed": False,
            "assumptions": "Only TCP command displacement is known; future visible points and span recovery are unknown."}
            for action in actions if action in ("up", "hold")],
        "note": "Current fragment bounds are not whole-object geometry. Reference span and completed identities are historical. Robot adjacency is a sensing clue, not causal occlusion or path safety. Completion only permits original final stability verification."}


def _placement_height_progress_context(snapshot, actions, step_m, *, prefix="placement_height_adjustment"):
    maximum_name, deficit_name = prefix + "_maximum_clearance_m", prefix + "_deficit_m"
    interval = placement_height_interval_v1(snapshot, maximum_measurement=maximum_name)
    value = None if interval is None else local_intent_interval_value_v1(interval, snapshot)
    mode = placement_relation_restore_mode_v1(snapshot)
    error = None if value is None else local_intent_interval_distance_v1(interval, value)
    evaluations = []
    if error is not None:
        for action in actions:
            direction = ((0.0, 0.0, 0.0) if action == "hold" else
                TRANSLATION_DIRECTIONS_ROBOT_BASE.get(AtomicAction(action)))
            if direction is None:
                continue
            delta = tuple(step_m * component for component in direction)
            projected = value + delta[2]
            projected_error = local_intent_interval_distance_v1(interval, projected)
            evaluations.append({"action": action, "expected_tcp_delta_robot_base_m": delta,
                "projected_subject_bottom_to_target_support_clearance_m": projected,
                "projected_local_error_m": projected_error, "projected_progress_m": error - projected_error,
                "source": "bounded_command_geometry", "is_observed": False,
                "assumptions": "subject follows TCP along Z; fixed target support height; ignores slip, dynamics, rotation and path collisions"})
    # 只比较同一阶段、动作前后均有效且仍处于持有/对齐作用域的原生测量。
    feedback = snapshot.phase_feedback
    previous_error = None
    basis = None if feedback is None else feedback.previous_observation_basis
    if error is not None and basis is not None and basis.same_scope_as(snapshot.skill):
        validity = dict(basis.measurement_evidence_current)
        old = {m.name: m for m in feedback.previous_measurements}
        names = ("subject_held", "placement_horizontal_relation_satisfied",
            deficit_name, maximum_name)
        action = snapshot.recent_action
        if (all(validity.get(name) is True and name in old for name in names)
            and old["subject_held"].value is True and old["placement_horizontal_relation_satisfied"].value is True
            and old[deficit_name].unit == "m"
            and old[deficit_name].source == "rgbd_geometry"
            and old[maximum_name].value == interval.maximum_m
            and action is not None and feedback.previous_action_epoch == action.start_action_epoch
            and snapshot.action_epoch == action.end_action_epoch):
            previous_error = old[deficit_name].value
    return {"status": "observed" if error is not None else "unknown" if mode in ("unknown", "placement") else "not_applicable",
        "scope": prefix + "_while_currently_held_and_horizontally_aligned",
        "interval": None if interval is None else asdict(interval), "clearance_m": value, "error_m": error,
        "local_condition_satisfied": None if error is None else error == 0.0,
        "previous_error_m": previous_error,
        "observed_progress_m": None if previous_error is None else previous_error - error,
        "prediction_available": error is not None, "action_evaluations": evaluations,
        "note": "Local height completion is separate from gripper opening and final placement. The interval never vetoes opening; commands predict geometry, not measured effects or safe paths."}


def build_local_intent_selection_context_v1(full_context):
    """侧面选择需要当前几何和全部目标对照；原子动作预测留给动作层。"""
    payload = copy.deepcopy(full_context)
    if "push_contact_geometry" in payload:
        payload["push_contact_geometry"] = [{"geometry": item["geometry"]}
            for item in payload["push_contact_geometry"]]
        payload["push_prediction_scope"] = "Intent selection sees all current sides and goal counterfactuals; per-action tool predictions are provided after selecting the local objective."
        if "push_goal_progress" in payload:
            payload["push_goal_progress"]["action_evaluations"] = []
    return payload


def _scope_push_goal_to_side(progress, side):
    """动作层聚焦已经选择的方向，不改变完整状态或意图层候选。"""
    result = copy.deepcopy(progress)
    direction = tuple(-v for v in side.outward_normal_xy)
    for geometry in [result["current"], *(item["translated_footprint_counterfactual"]
        for item in result.get("action_evaluations", ()) )]:
        if "directional_exit_requirements" in geometry:
            geometry["directional_exit_requirements"] = tuple(item for item in geometry["directional_exit_requirements"]
                if tuple(item["direction_xy"]) == direction)
    for item in result.get("action_evaluations", ()):
        change = item.get("native_goal_change")
        if change is not None and "directional_exit_remaining_reductions_m" in change:
            change["directional_exit_remaining_reductions_m"] = tuple(p for p in change["directional_exit_remaining_reductions_m"]
                if tuple(p["direction_xy"]) == direction)
    result["direction_scope"] = "selected_side_only; all side plans remain available on intent selection/replanning"
    return result


def build_local_intent_motor_context_v1(snapshot, candidate, full_context):
    """动作层只执行选定意图，不再次接收完整任务或其他候选。"""
    correction = local_intent_correction_v1(candidate, snapshot)
    selected = {
        "choice_id": candidate.choice_id, "objective": candidate.objective,
        "metric_kind": candidate.metric_kind, "reference_ref": candidate.reference_ref,
        "target_robot_base_m": candidate.target_robot_base_m,
        "correction_robot_base_m": correction,
        "axis_error_signs": None if correction is None else {
            axis: "positive" if correction[i] > 0 else "negative" if correction[i] < 0 else "zero"
            for i, axis in enumerate(("x", "y", "z"))},
        "distance_m": local_intent_metric_v1(candidate, snapshot),
        "completion_tolerance_m": candidate.tolerance_m,
        "maintain": candidate.maintain, "path_verified": candidate.path_verified,
        "contact_side": None if candidate.contact_side is None else asdict(candidate.contact_side),
    }
    preparing = candidate.choice_id.startswith(("prepare_push_side_edge_", "align_reprepared_push_edge_"))
    if preparing:
        selected["phase_requirement_roles"] = {
            "supported_push_contact": "restore_after_preparation; actual contact must be clear during reposition/alignment",
            "tcp_to_push_height_z_m": "restore_after_preparation; original contact height is not this local action target",
            "capability_goal_satisfied": "final_phase_verification; not this preparation step completion"}
    if candidate.completion_conditions:
        selected["completion_conditions"] = [{**asdict(condition), "satisfied": condition.satisfied}
            for condition in local_intent_conditions_v1(candidate, snapshot)]
    if candidate.progress_interval is not None:
        selected["progress_interval"] = asdict(candidate.progress_interval)
    evaluations = []
    if correction is not None and candidate.metric_kind in ("tcp_xy", "tcp_z"):
        # Predicted command geometry, not a physical/contact model or an action selector.
        current = local_intent_metric_v1(candidate, snapshot)
        steps = full_context["handle"].get("translation_step_by_action_m", {})
        for action in candidate.allowed_actions:
            direction = (0.0, 0.0, 0.0) if action == "hold" else TRANSLATION_DIRECTIONS_ROBOT_BASE.get(AtomicAction(action))
            if direction is None:
                continue
            step = steps.get(action, full_context["handle"]["translation_step_m"])
            if candidate.choice_id.startswith("prepare_grasp_view_"):
                step = grasp_translation_step_v1(snapshot, snapshot.skill.subject_ref, AtomicAction(action),
                    full_context["handle"]["translation_step_m"], candidate.target_robot_base_m)
            delta = tuple(step * value for value in direction)
            axes = (0, 1) if candidate.metric_kind == "tcp_xy" else (2,)
            projected = sqrt(sum((correction[i] - delta[i]) ** 2 for i in axes))
            evaluations.append({"action": action, "expected_tcp_delta_robot_base_m": delta,
                "projected_local_error_m": projected, "projected_progress_m": current - projected,
                "source": "bounded_command_geometry", "is_observed": False})
    if candidate.choice_id == "observe_grasp_geometry":
        for action in candidate.allowed_actions:
            step = grasp_translation_step_v1(snapshot, snapshot.skill.subject_ref, AtomicAction(action),
                full_context["handle"]["translation_step_m"])
            evaluations.append({"action": action, "expected_tcp_delta_robot_base_m": (0.,0.,step if action == "up" else 0.),
                "projected_visibility": None, "projected_local_error_m": None, "projected_progress_m": None,
                "source": "bounded_robot_retraction_geometry", "is_observed": False,
                "note": "Visibility improvement and complete geometry require the next fresh RGB-D observation."})
    contact_context = None
    if candidate.contact_side is not None and snapshot.skill.skill_kind == "surface_push":
        contact_context = next((item for item in full_context.get("push_contact_geometry", ())
            if item["geometry"].get("side", {}).get("edge_index") == candidate.contact_side.edge_index), None)
        if contact_context is None:
            contact_context = _contact_context(snapshot, candidate.contact_side, candidate.allowed_actions,
                full_context["handle"]["translation_step_m"])
        # 点准备以当前目标点为度量，不能被接触距离覆盖。
        if candidate.metric_kind not in ("tcp_xy", "tcp_z") and not candidate.choice_id.startswith("prepare_push_side_edge_"):
            evaluations = [item for item in contact_context["action_evaluations"] if item["action"] in candidate.allowed_actions]
    if candidate.choice_id.startswith("prepare_push_side_edge_"):
        values = current_runtime_measurements_v1(snapshot)
        height = values.get("tcp_to_push_reposition_height_z_m")
        clearance, tolerance = values.get("push_tool_above_subject_clearance_m"), values.get("push_height_tolerance_m")
        current = local_intent_metric_v1(candidate, snapshot)
        if height is not None and clearance is not None and tolerance is not None and current is not None:
            for action in candidate.allowed_actions:
                step = full_context["handle"]["translation_step_m"]
                delta = step if action == "up" else -step if action == "down" else 0.0
                height_deficit = max(0.0, abs(height - delta) - tolerance)
                projected = max(height_deficit, -clearance - delta, 0.0)
                evaluations.append({"action": action, "expected_tcp_delta_robot_base_m": (0.0, 0.0, delta),
                    "projected_reposition_height_deficit_m": height_deficit,
                    "projected_tool_above_subject_clearance_m": clearance + delta,
                    "projected_local_error_m": projected, "projected_progress_m": current - projected,
                    "is_observed": False, "source": "bounded_command_geometry",
                    "note": "Fixed subject and tool orientation; actual contact clear still requires fresh feedback."})
    if ((snapshot.skill.skill_kind == "transport" and snapshot.skill.phase == "align_target_xy"
        and candidate.choice_id == "align_target_xy") or candidate.choice_id == "restore_placement_relation"):
        evaluations = [item for item in full_context.get("transport_goal_progress", {}).get("action_evaluations", ())
            if item["action"] in candidate.allowed_actions]
    if candidate.choice_id == "retract_after_release":
        evaluations = [item for item in full_context.get("tool_transfer_clearance", {}).get("action_evaluations", ())
            if item["action"] in candidate.allowed_actions]
    if candidate.choice_id == "observe_task_geometry":
        evaluations = [item for item in full_context.get("goal_observation", {}).get("action_evaluations", ())
            if item["action"] in candidate.allowed_actions]
    if candidate.progress_interval is not None:
        height_key = ("placement_support_approach" if candidate.choice_id == "approach_placement_support"
            else "placement_height_adjustment")
        horizontal = {item["action"]: item for item in full_context.get("transport_goal_progress", {}).get("action_evaluations", ())}
        evaluations = [{**horizontal.get(item["action"], {}), **item}
            for item in full_context.get(height_key, {}).get("action_evaluations", ())
            if item["action"] in candidate.allowed_actions]
    phase = full_context["phase"]
    # 点意图也必须看到阶段条件；局部距离不替代完整阶段或持续约束。
    phase_state = copy.deepcopy(phase)
    phase_state["runtime_outcome"] = None if phase["verification"] is None else phase["verification"]["outcome"]
    names = {item["measurement"] for item in phase["conditions"]}
    names.update(item["condition"]["measurement"] for item in phase["maintain_constraints"]
                 if item["condition"] is not None)
    needs_goal = ("capability_goal_satisfied" in names or "placement_horizontal_relation_satisfied" in names
        or snapshot.skill.skill_kind in ("place", "surface_push"))
    if needs_goal:
        phase_state["bound_relation"] = full_context["subtask_definition"]["final_goal"]
    else:
        omitted = {"final_goal_relation", "capability_goal_satisfied"}
        phase_state["measurements"] = [item for item in phase["measurements"] if item["name"] not in omitted]
        phase_state["measurement_definitions"] = {key: value for key, value in phase["measurement_definitions"].items() if key not in omitted}
        phase_state["previous_measurements"] = [item for item in phase["previous_measurements"] if item["name"] not in omitted]
        for field in ("observation_basis", "previous_observation_basis"):
            basis = phase_state[field]
            if basis is not None:
                basis["measurement_evidence_current"] = {name: valid
                    for name, valid in basis["measurement_evidence_current"].items() if name not in omitted}
    intent = full_context["local_intent"]
    payload = {
        "schema": "JEVLocalActionContextV1", "schema_version": 1,
        "state_id": snapshot.state_id, "action_epoch": snapshot.action_epoch,
        "timestamp_s": snapshot.timestamp_s, "selected_local_intent": selected,
        "action_evaluations": evaluations,
        "contact_geometry": None if contact_context is None else contact_context["geometry"],
        "phase": phase_state,
        "local_intent": None if intent is None else {
            "execution": intent["execution"], "retained_approach": intent["retained_approach"],
            "handoff": intent["handoff"], "last_termination": intent["last_termination"],
            "push_contact_plan": intent["push_contact_plan"],
            "push_preparation": intent["push_preparation"],
            "push_goal_execution": intent["push_goal_execution"],
            "replan_count": intent["replan_count"], "blocked_reason": intent["blocked_reason"]},
        "robot": full_context["robot"], "physical": full_context["physical"],
        "subject": full_context["subject"], "dependencies": full_context["dependencies"],
        "target": full_context["target"] if needs_goal or candidate.metric_kind == "contact_geometry" else None,
        "recent_action": full_context["recent_action"],
        "decision_feedback": full_context["decision_feedback"],
        "observation": full_context["observation"],
        "handle": {**full_context["handle"], "available_actions": list(candidate.allowed_actions)},
        "release_geometry": full_context["interaction"]["release_geometry"],
    }
    if snapshot.press_event is not None:
        payload["press_event"] = asdict(snapshot.press_event)
    if snapshot.grasp_approach is not None:
        payload["grasp_approach"] = grasp_approach_context_v1(snapshot,include_options=False)
    if candidate.placement_preparation is not None:
        payload["placement_preparation"] = placement_preparation_context_v1(candidate,snapshot)
    if contact_context is not None and not preparing:
        payload["contact_action_evaluations"] = [item for item in contact_context["action_evaluations"] if item["action"] in candidate.allowed_actions]
        # 靠近工具不等于交互成立：显式区分几何修正与动作后响应采集。
        estimate = snapshot.physical.evidence.interaction_estimate
        values = current_runtime_measurements_v1(snapshot)
        current_estimate = (estimate is not None and
            (estimate.state_id, estimate.action_epoch, estimate.observation_id, estimate.subject_ref)
            == (snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id, snapshot.skill.subject_ref))
        acquisition = (current_estimate and estimate.status == "candidate"
            and all(values.get(name) is True for name in (
                "subject_supported", "grasp_not_held", "push_tool_closed_observed", "push_height_ready")))
        payload["contact_response_objective"] = {
            "motion_response_acquisition_needed": acquisition,
            "estimate_status": estimate.status if current_estimate else "unknown",
            "source": "current_rgbd_robot_interaction_estimate",
            "note": "When acquisition is needed, zero fixed-subject geometry error is not the action objective: compare permitted inward/tangential components and native goal counterfactuals, then inspect actual tool/object response. No direction is preselected and no contact/force is certified by a command."}
        if candidate.metric_kind in ("contact_geometry", "runtime_conditions"):
            # 同一预测的完整有限面诊断保留一次，避免重复放大模型输入。
            payload["action_evaluations"] = [{k: v for k, v in item.items()
                if k != "projected_contact_summary"} for item in evaluations]
    if "transport_goal_progress" in full_context:
        payload["transport_goal_progress"] = {key: value
            for key, value in full_context["transport_goal_progress"].items() if key != "action_evaluations"}
    if "push_goal_progress" in full_context:
        payload["push_goal_progress"] = ({**full_context["push_goal_progress"], "action_evaluations": [],
            "action_prediction_scope": "Final relation is deferred during selected preparation; use this local step's action_evaluations. All side-goal counterfactuals remain available for plan comparison."}
            if preparing else full_context["push_goal_progress"])
        clear_exit = full_context["push_goal_progress"]["current"].get("predicate") == "clear_of_region"
        if candidate.contact_side is not None and clear_exit:
            payload["push_goal_progress"] = _scope_push_goal_to_side(payload["push_goal_progress"], candidate.contact_side)
        payload["push_interaction"] = full_context["push_interaction"]
        payload["push_side_goal_counterfactuals"] = [{"side": item["side"],
            "exit_plan_geometry": item.get("exit_plan_geometry"),
            "native_goal_change": {key: value for key, value in (item.get("native_goal_change") or {}).items()
                if key != "directional_exit_remaining_reductions_m"},
            "is_observed": False, "is_physical_push_prediction": False}
            for item in full_context["push_side_goal_counterfactuals"]] if clear_exit else full_context["push_side_goal_counterfactuals"]
        if contact_context is None:
            # 没有选接触侧面的高度/推进目标仍看到全部当前几何，
            # 不重复携带每个未选侧面的原子动作预测。
            payload["push_contact_geometry"] = [{"geometry": item["geometry"]}
                for item in full_context["push_contact_geometry"]]
    if "placement_relation_restore" in full_context:
        payload["placement_relation_restore"] = full_context["placement_relation_restore"]
    if "placement_height_adjustment" in full_context:
        payload["placement_height_adjustment"] = {key: value
            for key, value in full_context["placement_height_adjustment"].items() if key != "action_evaluations"}
    if "placement_support_approach" in full_context:
        payload["placement_support_approach"] = {key: value
            for key, value in full_context["placement_support_approach"].items() if key != "action_evaluations"}
    if "tool_transfer_clearance" in full_context:
        payload["tool_transfer_clearance"] = {key: value
            for key, value in full_context["tool_transfer_clearance"].items() if key != "action_evaluations"}
    if "goal_observation" in full_context:
        payload["goal_observation"] = {key: value
            for key, value in full_context["goal_observation"].items() if key != "action_evaluations"}
    return payload


def _release_geometry(
    snapshot: StateSnapshotV2,
    subject: EntityV1 | None,
    target: EntityV1 | RegionV1 | None,
) -> dict[str, Any] | None:
    if snapshot.skill.skill_kind != "place":
        return None
    height = None
    if isinstance(target, RegionV1):
        evidence = target.geometry.surface_height_robot_base_m
        height = None if evidence is None else evidence.value
    elif isinstance(target, EntityV1):
        if snapshot.skill.goal is not None and snapshot.skill.goal.predicate is VisualPredicate.ON_SURFACE:
            height = target.geometry.top_center_robot_base_m.value
            height = None if height is None else height[2]
        elif target.support_region_ref is not None:
            region = next(
                (item for item in snapshot.visual_scene.regions
                 if item.region_id == target.support_region_ref),
                None,
            )
            evidence = None if region is None else region.geometry.surface_height_robot_base_m
            height = None if evidence is None else evidence.value
    center = None if subject is None else subject.geometry.centroid_robot_base_m.value
    extent = None if subject is None else subject.geometry.extent_m.value
    bottom = None if center is None or extent is None else center[2] - 0.5 * extent[2]
    tcp_z = snapshot.robot.tcp_position_robot_base_m[2]
    return {
        "observable": height is not None and bottom is not None,
        "target_support_height_robot_base_m": height,
        "tcp_to_target_support_vertical_m": None if height is None else tcp_z - height,
        "subject_bottom_to_target_support_clearance_m": (
            None if height is None or bottom is None else bottom - height
        ),
        "subject_bottom_robot_base_m": bottom,
    }


def build_jev_decision_context_v1(
    snapshot: StateSnapshotV2,
    *,
    translation_step_m: float,
    grasp_xy_tolerance_m: float,
) -> JevDecisionContextV1:
    snapshot = apply_placement_action_permissions_v1(snapshot)
    if not isfinite(translation_step_m) or translation_step_m <= 0.0:
        raise ValueError("translation_step_m must be finite and positive")
    if not isfinite(grasp_xy_tolerance_m) or grasp_xy_tolerance_m <= 0.0:
        raise ValueError("grasp_xy_tolerance_m must be finite and positive")
    subject = next(
        (
            entity
            for entity in snapshot.visual_scene.entities
            if entity.track_id == snapshot.skill.subject_ref
        ),
        None,
    )
    target = _referenced_target(snapshot)
    interaction = snapshot.interaction
    recent = snapshot.recent_action
    goal = snapshot.skill.goal
    contact = snapshot.physical.effector_contact
    support = snapshot.physical.support_state
    is_pick_xy = (
        snapshot.skill.skill_kind == "pick"
        and snapshot.skill.phase == "align_subject_xy"
    )
    definition = _subtask_definition(snapshot)
    phase = next((item for item in definition.phases if item[0] == snapshot.skill.phase), None)
    role, goal_delta, tolerance = _motion_error(snapshot, subject, target)
    if is_pick_xy and tolerance is None:
        tolerance = grasp_xy_tolerance_m
    distance = None if goal_delta is None else sqrt(sum(value * value for value in goal_delta))
    dominant_axis = (None if goal_delta is None or distance == 0.0 else
                     ("x", "y", "z")[max(range(3), key=lambda index: abs(goal_delta[index]))])
    axes = [] if phase is None else list(phase[2].active_axes)
    centroid = (None if subject is None or subject.visibility is not Visibility.CLEAR else
                subject.geometry.centroid_robot_base_m.value)
    subject_offset = None if centroid is None else [
        centroid[index] - snapshot.robot.tcp_position_robot_base_m[index] for index in range(3)]
    feedback = snapshot.phase_feedback
    diagnostic = snapshot.physical.evidence.grasp_diagnostic
    diagnostic_bound = diagnostic is not None and (
        diagnostic.subject_ref == snapshot.skill.subject_ref
        and diagnostic.based_on_state_id == snapshot.state_id
        and diagnostic.based_on_action_epoch == snapshot.action_epoch
        and diagnostic.based_on_observation_id == snapshot.visual_scene.observation_id
        and diagnostic.timestamp_s == snapshot.timestamp_s)

    payload = {
        "schema": "JEVDecisionContextV1",
        "schema_version": 1,
        "state_id": snapshot.state_id,
        "action_epoch": snapshot.action_epoch,
        "timestamp_s": snapshot.timestamp_s,
        "subtask_definition": _definition_payload(definition),
        "local_intent": None if snapshot.local_intent is None else {
            **asdict(snapshot.local_intent),
            "execution_correction_robot_base_m": (
                None if snapshot.local_intent.execution is None else
                local_intent_correction_v1(snapshot.local_intent.execution.candidate, snapshot)),
        },
        "phase": {
            "phase_id": snapshot.skill.phase,
            "binding": {"skill_instance_id": snapshot.skill.skill_instance_id,
                "subject_ref": snapshot.skill.subject_ref, "source_ref": snapshot.skill.source_ref,
                "target_ref": snapshot.skill.target_ref},
            "completion_contract": None if phase is None else phase[1],
            "completion_authority": "runtime_verifier",
            "condition_role": "completion_goals_not_action_preconditions",
            "constraint_role": "scoped_evidence_reports_not_new_action_guards",
            "definition_ref": definition.instance_id,
            "definition": None if phase is None else asdict(phase[2]),
            "measurement_definitions": {} if feedback is None else {
                item.name: PHASE_MEASUREMENT_DEFINITIONS_V1[item.name]
                for item in feedback.measurements},
            "conditions": [] if feedback is None else [
                {**asdict(item), "satisfied": item.satisfied} for item in feedback.conditions],
            "maintain_constraints": [] if feedback is None else [{**asdict(item), "condition": (
                None if item.condition is None else {**asdict(item.condition), "satisfied": item.condition.satisfied})}
                for item in feedback.maintain_constraints],
            "evidence_basis": None if feedback is None or feedback.evidence_basis is None else asdict(feedback.evidence_basis),
            "observation_basis": None if feedback is None else _observation_basis_payload(feedback.observation_basis, current=True),
            "previous_observation_basis": None if feedback is None else _observation_basis_payload(feedback.previous_observation_basis),
            "configured_parameters": {} if feedback is None else dict(feedback.parameters),
            "verification": None if feedback is None else {
                "outcome": feedback.outcome, "reason": feedback.reason,
                "based_on_state_id": feedback.based_on_state_id,
                "based_on_action_epoch": feedback.based_on_action_epoch,
                "based_on_observation_id": feedback.based_on_observation_id,
                "consecutive_pass_count": feedback.consecutive_pass_count,
                "required_consecutive_passes": feedback.required_consecutive_passes,
            },
            "measurements": [] if feedback is None else [
                {**asdict(item), "status": "unobservable" if item.value is None else
                 "measured" if item.source == "robot_feedback" else
                 "commanded" if item.source == "command" else
                 "configured" if item.source == "deployment_configuration" else "derived"}
                for item in feedback.measurements],
            "previous_measurements": [] if feedback is None else [asdict(item) for item in feedback.previous_measurements],
            "previous_state_id": None if feedback is None else feedback.previous_state_id,
            "previous_action_epoch": None if feedback is None else feedback.previous_action_epoch,
        },
        "task": {
            "task_kind": snapshot.task.task_kind.value,
        },
        "capability": {
            "kind": snapshot.skill.skill_kind,
            "runtime_checkpoint": snapshot.skill.phase,
            "subject_ref": snapshot.skill.subject_ref,
            "target_ref": snapshot.skill.target_ref,
            "source_ref": snapshot.skill.source_ref,
            "goal": (
                None
                if goal is None
                else {
                    "predicate": goal.predicate.value,
                    "reference_ref": goal.reference_ref,
                    "desired_value": goal.desired_value,
                }
            ),
        },
        "robot": {
            "tcp_position_robot_base_m": list(
                snapshot.robot.tcp_position_robot_base_m
            ),
            "tcp_orientation_robot_base_wxyz": list(snapshot.robot.tcp_orientation_robot_base_wxyz),
            "gripper": {
                "opening_m": snapshot.robot.gripper.opening_m,
                "target_opening_m": snapshot.robot.gripper.target_opening_m,
                "motion_state": snapshot.robot.gripper.motion_state.value,
            },
            "execution_state": snapshot.robot.execution_state.value,
            "feedback_timestamp_s": snapshot.sources.robot_measurement_timestamp_s,
            "right_finger_joint_position_m": snapshot.robot.gripper.right_joint_position_m,
            "tool_calibration": None if snapshot.robot.tool_geometry is None else {
                "source": snapshot.robot.tool_geometry.source,
                "parts": [p.part_id for p in snapshot.robot.tool_geometry.parts]},
        },
        "subject": None if subject is None else _entity(subject),
        "target": (
            None
            if target is None
            else _entity(target)
            if isinstance(target, EntityV1)
            else _region(target)
        ),
        "dependencies": _dependencies(snapshot, subject, target),
        "interaction": {
            "target_relation": (
                None
                if interaction.local_goal is None
                else interaction.local_goal.target_relation
            ),
            "goal_delta_robot_base_m": goal_delta,
            "distance_m": distance,
            "dominant_axis": dominant_axis,
            "active_axes": axes,
            "completion_tolerance_m": tolerance,
            "motion_error_role": role,
            "motion_error_observable": goal_delta is not None,
            "subject_from_tcp_robot_base_m": subject_offset,
            "subject_offset_role": "observation_only_not_a_final_goal_error",
            "release_geometry": _release_geometry(snapshot, subject, target),
        },
        "physical": {
            "grasp_state": snapshot.physical.grasp_state.value,
            "held_object_ref": snapshot.physical.held_object_ref,
            "grasp_probe_lifted": snapshot.physical.evidence.grasp_probe_lifted,
            "grasp_probe_failed": snapshot.physical.evidence.grasp_probe_failed,
            "interaction_estimate": (None if snapshot.physical.evidence.interaction_estimate is None
                else asdict(snapshot.physical.evidence.interaction_estimate)),
            "grasp_diagnostic": asdict(diagnostic) if diagnostic_bound else None,
            "grasp_diagnostic_current": bool(diagnostic_bound and feedback is not None
                and feedback.evidence_basis is not None
                and feedback.evidence_basis.visual_evidence_current
                and feedback.evidence_basis.robot_evidence_current),
            "evidence_sources": {
                "contact": snapshot.physical.evidence.contact_source,
                "grasp": snapshot.physical.evidence.grasp_source,
                "support": snapshot.physical.evidence.support_source,
            },
            "contact": {
                "mode": contact.mode.value,
                "subject_ref": contact.subject_ref,
                "left_finger": None if contact.mode.value == "unknown" else contact.left_finger_contact,
                "right_finger": None if contact.mode.value == "unknown" else contact.right_finger_contact,
                "support": contact.support_contact,
            },
            "support": {
                "subject_ref": support.subject_ref,
                "region_ref": support.support_region_ref,
                "supported": support.supported,
            },
        },
        "recent_action": (
            None
            if recent is None
            else {
                "action": recent.action,
                "status": recent.status,
                "decision_id": recent.decision_id,
                "start_action_epoch": recent.start_action_epoch,
                "end_action_epoch": recent.end_action_epoch,
                "commanded_delta_robot_base_m": list(
                    recent.commanded_delta_robot_base_m
                ),
                "measured_delta_robot_base_m": list(
                    recent.measured_delta_robot_base_m
                ),
            }
        ),
        "decision_feedback": (
            None
            if snapshot.decision_feedback is None
            or snapshot.decision_feedback.phase != snapshot.skill.phase
            else {
                "action": snapshot.decision_feedback.action,
                "status": snapshot.decision_feedback.status,
                "reason": snapshot.decision_feedback.reason,
                "error_before_m": snapshot.decision_feedback.error_before_m,
                "projected_error_after_m": snapshot.decision_feedback.projected_error_after_m,
                "error_after_m": snapshot.decision_feedback.error_after_m,
                "progress_m": snapshot.decision_feedback.progress_m,
            }
        ),
        "observation": {
            "observation_id": snapshot.visual_scene.observation_id,
            "capture_timestamp_s": snapshot.visual_scene.capture_timestamp_s,
            "sensor_id": snapshot.visual_scene.sensor.sensor_id,
            "calibration_id": snapshot.visual_scene.sensor.calibration_id,
            "visual_is_fresh": snapshot.freshness.visual_is_fresh,
            "visual_age_s": snapshot.freshness.visual_age_s,
        },
        "handle": {
            "translation_step_m": translation_step_m,
            "available_actions": list(snapshot.skill.allowed_actions),
            "directions_robot_base": {
                action: direction
                for action, direction in {
                    "forward": "+x",
                    "backward": "-x",
                    "left": "+y",
                    "right": "-y",
                    "up": "+z",
                    "down": "-z",
                }.items()
                if action in snapshot.skill.allowed_actions
            },
        },
    }
    if feedback is not None and feedback.released_placement_outcome is not None:
        released = feedback.released_placement_outcome
        payload["phase"]["released_placement_outcome"] = {
            "status": released.status, "reason": released.reason,
            "goal_satisfied": released.goal_satisfied, "support_ref": released.support_ref,
            "stable_miss_observation_count": len(released.stable_miss_samples),
            "required_observations": released.required_observations,
            "waiting_started_s": released.waiting_started_s,
            "waiting_remaining_s": released.waiting_remaining_s,
            "release_evidence": {"state_id": released.release_state_id,
                "action_epoch": released.release_action_epoch,
                "observation_id": released.release_observation_id,
                "timestamp_s": released.release_timestamp_s},
            "stable_miss_evidence": [{"state_id": s.state_id, "action_epoch": s.action_epoch,
                "observation_id": s.observation_id, "capture_timestamp_s": s.capture_timestamp_s}
                for s in released.stable_miss_samples],
            "note": "Runtime classifies release outcome; goal_candidate is not final success, unknown is not a miss. Waiting and historical release evidence do not select an action or authorize a retry.",
        }
    if snapshot.press_event is not None:
        payload["press_event"] = asdict(snapshot.press_event)
    if snapshot.grasp_approach is not None:
        payload["grasp_approach"] = grasp_approach_context_v1(snapshot)
    if snapshot.skill.skill_kind == "surface_push" and snapshot.skill.phase in ("align_precontact", "establish_contact", "push_toward_goal"):
        sides = observed_contact_sides_v1(snapshot)
        context = snapshot.local_intent
        selected = None if context is None else (
            context.execution.candidate if context.execution is not None and context.execution.status == "active"
            else None if context.handoff is None else context.handoff.candidate)
        if selected is not None and selected.contact_side is not None:
            sides = tuple(side for side in sides if side.edge_index == selected.contact_side.edge_index)
        elif context is not None and context.push_contact_plan is not None and push_interaction_mode_v1(snapshot) == "advance":
            side = current_push_plan_side_v1(context.push_contact_plan, snapshot)
            sides = () if side is None else (side,)
        payload["push_contact_geometry"] = [_contact_context(snapshot, side,
            snapshot.skill.allowed_actions, translation_step_m) for side in sides]
        payload["contact_geometry_note"] = (
            "Public finite-side estimates; model chooses the side/action. A zero gap is not contact confirmation. "
            "Missing complete geometry is unknown, never zero or an old footprint presented as a new observation.")
        payload["push_goal_progress"] = _push_goal_context(snapshot, snapshot.skill.allowed_actions, translation_step_m)
        # 当前所有侧面均展示，不受活动侧面筛选，也不排名或选择。
        payload["push_side_goal_counterfactuals"] = []
        for side in observed_contact_sides_v1(snapshot):
            delta = tuple(-translation_step_m * v for v in side.outward_normal_xy)
            relation = _push_goal_counterfactual(snapshot, delta)
            payload["push_side_goal_counterfactuals"].append({"side": asdict(side),
                "assumed_subject_delta_xy_m": delta, "translated_footprint_counterfactual": relation,
                "exit_plan_geometry": next((item for item in payload["push_goal_progress"]["current"].get(
                    "directional_exit_requirements", ()) if item["edge_index"] == side.edge_index), None),
                "native_goal_change": _native_push_goal_change(payload["push_goal_progress"]["current"], relation),
                "is_observed": False, "is_physical_push_prediction": False,
                "source": "public_footprint_translation_counterfactual"})
        payload["push_interaction"] = {"mode": push_interaction_mode_v1(snapshot),
            "current_measurements": current_runtime_measurements_v1(snapshot),
            "side_plan_status": "historical_model_selection" if context is not None and context.push_contact_plan is not None else "unknown",
            "preparation": None if context is None or context.push_preparation is None else asdict(context.push_preparation),
            "selected_face_contact_verification": "unsupported", "path_safety": "unsupported",
            "note": "Body contact is current evidence, not selected-face verification. Contact and original height are restoration goals when false, not restoration prerequisites. Missing/unknown prerequisites allow observation only; explicitly false support/closure/not-held prerequisites block pushing."}
    restore_mode = placement_relation_restore_mode_v1(snapshot)
    if snapshot.skill.skill_kind == "place" and snapshot.skill.phase == "observe_task_geometry":
        payload["goal_observation"] = _task_observation_context(
            snapshot, snapshot.skill.allowed_actions, translation_step_m)
    if snapshot.skill.skill_kind == "place" and snapshot.skill.phase == "retract_after_release":
        payload["tool_transfer_clearance"] = _tool_transfer_progress_context(
            snapshot, snapshot.skill.allowed_actions, translation_step_m)
    if restore_mode is not None:
        payload["placement_height_adjustment"] = _placement_height_progress_context(
            snapshot, snapshot.skill.allowed_actions, translation_step_m)
        payload["placement_support_approach"] = _placement_height_progress_context(
            snapshot, snapshot.skill.allowed_actions, translation_step_m, prefix="placement_support_approach")
        payload["placement_relation_restore"] = {
            "mode": restore_mode,
            "correction_flow": None if snapshot.placement_correction is None else asdict(snapshot.placement_correction),
            "transport_clearance_requirement": {
                "scope": "initiated_clearance_correction_and_any_horizontal_preparation",
                "status": ("required" if restore_mode == "restore_clearance" else
                    "satisfied" if restore_mode == "restore_relation" else
                    "unknown" if restore_mode == "unknown" else "not_applicable"),
                "note": "A started clearance correction ends only after freshly reaching its original clearance goal. Its record is historical flow state, not fresh holding/clearance proof. Outside this obligation, the transport threshold is not a normal descent height goal.",
            },
            "current_measurements": current_runtime_measurements_v1(snapshot),
            "available_actions": list(apply_placement_action_permissions_v1(snapshot).skill.allowed_actions),
            "path_safety": "unsupported",
            "note": "Current evidence, not historical transport completion. Restore clearance before horizontal correction; a restored relation completes only the local intent, not release or placement verification.",
        }
    if ((snapshot.skill.skill_kind == "transport" and snapshot.skill.phase == "align_target_xy")
        or restore_mode in ("restore_relation", "placement")):
        payload["transport_goal_progress"] = _transport_progress_context(
            snapshot, snapshot.skill.allowed_actions, translation_step_m)
    local = payload["local_intent"]
    if local is not None:
        entries = list(local["candidates"])
        if local["execution"] is not None:
            entries.append(local["execution"]["candidate"])
        if local["last_termination"] is not None:
            entries.append(local["last_termination"]["execution"]["candidate"])
        for item in entries:
            evidence = item.get("placement_preparation")
            if evidence is not None:
                evidence.pop("region_boundary_xy_m",None)
        execution = None if snapshot.local_intent is None else snapshot.local_intent.execution
        if execution is not None and execution.candidate.placement_preparation is not None:
            payload["placement_preparation"] = placement_preparation_context_v1(execution.candidate,snapshot)
    return JevDecisionContextV1(payload)
