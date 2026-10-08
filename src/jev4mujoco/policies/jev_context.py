from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any
from jev4mujoco.contracts.actions import AtomicAction, TRANSLATION_DIRECTIONS_ROBOT_BASE

from jev4mujoco.runtime.progress import pick_xy_error_v1
from jev4mujoco.runtime.grasp_geometry import grasp_pose_correction_v1
from jev4mujoco.runtime.local_intent import local_intent_correction_v1, local_intent_metric_v1
from jev4mujoco.runtime.tool_geometry import contact_geometry_v1, observed_contact_sides_v1
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2
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
    for action in actions:
        direction = ((0.0, 0.0, 0.0) if action == "hold" else
            TRANSLATION_DIRECTIONS_ROBOT_BASE.get(AtomicAction(action)))
        if direction is None:
            continue
        delta = tuple(step_m * value for value in direction)
        predicted = contact_geometry_v1(snapshot, side, delta)
        projected = predicted["local_error_m"]
        evaluations.append({"action": action, "expected_tcp_delta_robot_base_m": delta,
            "projected_local_error_m": projected,
            "projected_progress_m": None if current is None or projected is None else current - projected,
            "projected_contact_summary": _contact_prediction_summary(predicted), "source": "bounded_command_geometry",
            "is_observed": False, "assumptions": "fixed subject and tool orientation; ignores dynamics and path collisions"})
    return {"geometry": geometry, "action_evaluations": evaluations}


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
            delta = tuple(step * value for value in direction)
            axes = (0, 1) if candidate.metric_kind == "tcp_xy" else (2,)
            projected = sqrt(sum((correction[i] - delta[i]) ** 2 for i in axes))
            evaluations.append({"action": action, "expected_tcp_delta_robot_base_m": delta,
                "projected_local_error_m": projected, "projected_progress_m": current - projected,
                "source": "bounded_command_geometry", "is_observed": False})
    contact_context = None
    if candidate.metric_kind == "contact_geometry":
        contact_context = next((item for item in full_context.get("push_contact_geometry", ())
            if item["geometry"].get("side", {}).get("edge_index") == candidate.contact_side.edge_index), None)
        if contact_context is None:
            contact_context = _contact_context(snapshot, candidate.contact_side, candidate.allowed_actions,
                full_context["handle"]["translation_step_m"])
        evaluations = contact_context["action_evaluations"]
    phase = full_context["phase"]
    # 点意图也必须看到阶段条件；局部距离不替代完整阶段或持续约束。
    phase_state = copy.deepcopy(phase)
    phase_state["runtime_outcome"] = None if phase["verification"] is None else phase["verification"]["outcome"]
    names = {item["measurement"] for item in phase["conditions"]}
    names.update(item["condition"]["measurement"] for item in phase["maintain_constraints"]
                 if item["condition"] is not None)
    needs_goal = ("capability_goal_satisfied" in names or "placement_horizontal_relation_satisfied" in names
        or snapshot.skill.skill_kind == "place")
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
    return {
        "schema": "JEVLocalActionContextV1", "schema_version": 1,
        "state_id": snapshot.state_id, "action_epoch": snapshot.action_epoch,
        "timestamp_s": snapshot.timestamp_s, "selected_local_intent": selected,
        "action_evaluations": evaluations,
        "contact_geometry": None if contact_context is None else contact_context["geometry"],
        "phase": phase_state,
        "local_intent": None if intent is None else {
            "execution": intent["execution"], "retained_approach": intent["retained_approach"],
            "handoff": intent["handoff"], "last_termination": intent["last_termination"],
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
    if snapshot.skill.skill_kind == "surface_push" and snapshot.skill.phase == "establish_contact":
        sides = observed_contact_sides_v1(snapshot)
        context = snapshot.local_intent
        selected = None if context is None else (
            context.execution.candidate if context.execution is not None and context.execution.status == "active"
            else None if context.handoff is None else context.handoff.candidate)
        if selected is not None and selected.contact_side is not None:
            sides = tuple(side for side in sides if side.edge_index == selected.contact_side.edge_index)
        payload["push_contact_geometry"] = [_contact_context(snapshot, side,
            snapshot.skill.allowed_actions, translation_step_m) for side in sides]
        payload["contact_geometry_note"] = (
            "Public finite-side estimates; model chooses the side/action. A zero gap is not contact confirmation. "
            "Missing complete geometry is unknown, never zero or an old footprint presented as a new observation.")
    return JevDecisionContextV1(payload)
