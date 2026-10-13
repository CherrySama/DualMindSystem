"""推动的当前条件与公开关系几何；不选择侧面或动作，不预测动力学。"""
from jev4mujoco.contracts.visual_scene import Visibility, VisualPredicate
from jev4mujoco.perception.surface_geometry import (
    signed_point_polygon_clearance_v1, polygons_overlap_v1,
    polygon_clearance_v1, footprint_clear_of_region_v1,
    polygon_contains_polygon_v1,
)
from jev4mujoco.runtime.tool_geometry import observed_contact_sides_v1
from jev4mujoco.runtime.push_exit_geometry import directional_exit_geometry_v1
from math import hypot


def push_reposition_ready_v1(snapshot):
    from jev4mujoco.runtime.local_intent import current_runtime_measurements_v1
    values = current_runtime_measurements_v1(snapshot)
    deficit = values.get("push_reposition_height_deficit_m")
    clearance = values.get("push_tool_above_subject_clearance_m")
    if deficit is None or clearance is None or values.get("tool_subject_contact_clear") is None:
        return None
    return (deficit == 0.0 and clearance > 0.0
        and all(values.get(name) is True for name in (
            "tool_subject_contact_clear", "subject_supported", "grasp_not_held", "push_tool_closed_observed")))


def push_outer_side_point_v1(snapshot, side):
    """每帧从当前公开边重建外侧点，保留原边界与距离条件。"""
    from jev4mujoco.runtime.local_intent import current_runtime_measurements_v1
    values = current_runtime_measurements_v1(snapshot)
    clearance, tolerance = values.get("push_precontact_clearance_m"), values.get("push_precontact_xy_tolerance_m")
    if clearance is None or tolerance is None:
        return None
    point = tuple((side.start_xy_m[i] + side.end_xy_m[i]) / 2 + side.outward_normal_xy[i] * clearance for i in (0, 1))
    source = next((r for r in snapshot.visual_scene.regions if r.region_id == snapshot.skill.source_ref), None)
    subject = next((e for e in snapshot.visual_scene.entities if e.track_id == snapshot.skill.subject_ref), None)
    boundary = None if source is None or source.geometry.boundary_xy_robot_base_m is None else source.geometry.boundary_xy_robot_base_m.value
    footprint = None if subject is None or subject.visibility is not Visibility.CLEAR else subject.geometry.footprint_xy_robot_base_m.value
    if (boundary is None or footprint is None or signed_point_polygon_clearance_v1(point, boundary) > 0
        or abs(signed_point_polygon_clearance_v1(point, footprint) - clearance) > tolerance):
        return None
    return (*point, snapshot.robot.tcp_position_robot_base_m[2])


def push_side_aligned_v1(snapshot, side):
    from jev4mujoco.runtime.local_intent import current_runtime_measurements_v1
    point = push_outer_side_point_v1(snapshot, side)
    tolerance = current_runtime_measurements_v1(snapshot).get("push_precontact_xy_tolerance_m")
    return point is not None and tolerance is not None and hypot(
        point[0] - snapshot.robot.tcp_position_robot_base_m[0],
        point[1] - snapshot.robot.tcp_position_robot_base_m[1]) <= tolerance


def current_push_plan_side_v1(plan, snapshot):
    if plan is None or not plan.matches(snapshot.skill):
        return None
    # 只接受当前观测中可对应的轮廓边；平移不要求旧端点位置保持。
    return next((side for side in observed_contact_sides_v1(snapshot)
        if side.edge_index == plan.selected_side.edge_index
        and sum(a * b for a, b in zip(side.outward_normal_xy,
            plan.selected_side.outward_normal_xy)) >= 1 - 1e-6), None)


def push_interaction_mode_v1(snapshot):
    from jev4mujoco.runtime.local_intent import current_runtime_measurements_v1
    if snapshot.skill.skill_kind != "surface_push" or snapshot.skill.phase != "push_toward_goal":
        return None
    values = current_runtime_measurements_v1(snapshot)
    prerequisites = (values.get("subject_supported"), values.get("grasp_not_held"),
        values.get("push_tool_closed_observed"))
    if any(value is False for value in prerequisites):
        return "blocked"
    subject = next((e for e in snapshot.visual_scene.entities if e.track_id == snapshot.skill.subject_ref), None)
    if (any(value is not True for value in prerequisites)
        or snapshot.robot.tool_geometry is None
        or snapshot.robot.gripper.right_joint_position_m is None
        or subject is None or subject.visibility is not Visibility.CLEAR
        or subject.geometry.centroid_robot_base_m.value is None or subject.geometry.extent_m.value is None
        or not observed_contact_sides_v1(snapshot)):
        return "unknown"
    contact, height = values.get("supported_push_contact"), values.get("push_height_ready")
    if contact is None or height is None:
        return "unknown"
    return "advance" if contact is True and height is True else "restore_contact"


def push_goal_geometry_v1(snapshot, *, delta_xy=(0.0, 0.0), evidence_current=None, parameters=None):
    """当前公开轮廓或轮廓平移反事实；各量保持原生单位。"""
    from jev4mujoco.runtime.local_intent import current_runtime_measurements_v1
    if evidence_current is None:
        evidence_current = bool(current_runtime_measurements_v1(snapshot))
    goal = snapshot.skill.goal
    unknown = {"status": "unknown", "goal_satisfied": None}
    if not evidence_current or goal is None:
        return unknown
    subject = next((e for e in snapshot.visual_scene.entities if e.track_id == snapshot.skill.subject_ref), None)
    region = next((r for r in snapshot.visual_scene.regions if r.region_id == goal.reference_ref), None)
    footprint = None if subject is None or subject.visibility is not Visibility.CLEAR else subject.geometry.footprint_xy_robot_base_m.value
    boundary = None if region is None or region.geometry.boundary_xy_robot_base_m is None else region.geometry.boundary_xy_robot_base_m.value
    if footprint is None or boundary is None:
        return unknown
    footprint = tuple((float(x + delta_xy[0]), float(y + delta_xy[1])) for x, y in footprint)
    distances = tuple(float(signed_point_polygon_clearance_v1(p, boundary)) for p in footprint)
    result = {"status": "observed", "predicate": goal.predicate.value,
        "vertex_signed_boundary_distances_m": distances}
    if goal.predicate is VisualPredicate.IN_REGION:
        satisfied = bool(polygon_contains_polygon_v1(boundary, footprint))
        result.update(vertex_outside_distances_m=tuple(max(0.0, d) for d in distances),
            containment_margin_m=-max(distances))
    elif goal.predicate is VisualPredicate.CLEAR_OF_REGION:
        parameters = dict(snapshot.phase_feedback.parameters) if parameters is None else parameters
        required = parameters.get("surface_push_clear_of_region_margin_m")
        if required is None:
            return unknown
        overlap = bool(polygons_overlap_v1(footprint, boundary))
        clearance = float(polygon_clearance_v1(footprint, boundary))
        satisfied = bool(footprint_clear_of_region_v1(footprint, boundary, required))
        result.update(footprints_overlap=overlap, footprint_clearance_m=clearance,
            required_clearance_m=required, clearance_deficit_m=max(0.0, required - clearance),
            exit_diagnostic="Signed vertex distances show boundary crossing; clearance is zero while overlapping. These are separate diagnostics, not a combined progress distance.")
        source = next((r for r in snapshot.visual_scene.regions if r.region_id == snapshot.skill.source_ref), None)
        support = None if source is None or source.geometry.boundary_xy_robot_base_m is None else source.geometry.boundary_xy_robot_base_m.value
        result["exit_region_ref"] = region.region_id
        result["exit_region_boundary_xy_m"] = boundary
        result["directional_exit_requirements"] = tuple({"edge_index": side.edge_index,
            **directional_exit_geometry_v1(footprint, boundary,
                tuple(-v for v in side.outward_normal_xy), required, support)}
            for side in observed_contact_sides_v1(snapshot))
    else:
        return {"status": "unsupported", "goal_satisfied": None}
    result["goal_satisfied"] = satisfied == goal.desired_value
    return result
