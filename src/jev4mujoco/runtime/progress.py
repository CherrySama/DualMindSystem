from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from jev4mujoco.contracts.actions import AtomicAction, TRANSLATION_DIRECTIONS_ROBOT_BASE
from jev4mujoco.contracts.state_snapshot import PhaseMeasurementV1, StateSnapshotV2
from jev4mujoco.contracts.visual_scene import Visibility, VisualPredicate
from jev4mujoco.perception.relations import _bounds, _xy_overlap_fraction
from jev4mujoco.perception.surface_geometry import (
    polygon_clearance_v1, signed_point_polygon_clearance_v1,
)


def placement_horizontal_measurements_v1(
    snapshot: StateSnapshotV2, parameters: dict[str, float], *,
    evidence_current: bool, delta_xy: tuple[float, float] = (0.0, 0.0),
    predicted: bool = False,
) -> tuple[PhaseMeasurementV1, ...]:
    """Relation deficits in native units, using the registered horizontal semantics.

    Predictions translate the public footprint rigidly; they do not certify
    holding, actual motion, path safety, Z, support or eventual placement.
    """
    goal = snapshot.skill.goal
    if goal is None or not goal.desired_value:
        return ()  # No invented inverse relation or arbitrary goal point.
    predicate = goal.predicate
    names = {
        VisualPredicate.IN_REGION: (("placement_vertex_signed_clearances_m", "m"),
            ("placement_vertex_outside_distances_m", "m"), ("placement_containment_margin_m", "m")),
        VisualPredicate.INSIDE: (("placement_vertex_signed_clearances_m", "m"),
            ("placement_vertex_outside_distances_m", "m"), ("placement_containment_margin_m", "m")),
        VisualPredicate.LEFT_OF: (("placement_directional_gap_m", "m"),
            ("placement_directional_gap_deficit_m", "m")),
        VisualPredicate.RIGHT_OF: (("placement_directional_gap_m", "m"),
            ("placement_directional_gap_deficit_m", "m")),
        VisualPredicate.IN_FRONT_OF: (("placement_directional_gap_m", "m"),
            ("placement_directional_gap_deficit_m", "m")),
        VisualPredicate.NEAR: (("placement_footprint_clearance_m", "m"),
            ("placement_clearance_deficit_m", "m")),
        VisualPredicate.SEPARATED: (("placement_footprint_clearance_m", "m"),
            ("placement_clearance_deficit_m", "m")),
        VisualPredicate.ON_SURFACE: (("placement_overlap_fraction", "fraction"),
            ("placement_overlap_fraction_deficit", "fraction"),
            ("placement_overlap_fraction_margin", "fraction"),
            ("placement_support_approach_clearance_m", "m")),
    }.get(predicate, ())
    values = {}
    subject = next((e for e in snapshot.visual_scene.entities
        if e.track_id == snapshot.skill.subject_ref), None)
    footprint = (None if subject is None or subject.visibility is not Visibility.CLEAR
        else subject.geometry.footprint_xy_robot_base_m.value)
    if evidence_current and footprint is not None:
        footprint = tuple((x + delta_xy[0], y + delta_xy[1]) for x, y in footprint)
        if predicate in (VisualPredicate.IN_REGION, VisualPredicate.INSIDE):
            region = next((r for r in snapshot.visual_scene.regions
                if r.region_id == goal.reference_ref), None)
            boundary = (None if region is None or region.geometry.boundary_xy_robot_base_m is None
                else region.geometry.boundary_xy_robot_base_m.value)
            if boundary is not None:
                distances = tuple(signed_point_polygon_clearance_v1(p, boundary) for p in footprint)
                values["placement_vertex_signed_clearances_m"] = distances
                values["placement_vertex_outside_distances_m"] = tuple(max(0.0, d) for d in distances)
                values["placement_containment_margin_m"] = -max(distances)
        else:
            reference = next((e for e in snapshot.visual_scene.entities
                if e.track_id == goal.reference_ref), None)
            other = (None if reference is None or reference.visibility is not Visibility.CLEAR
                else reference.geometry.footprint_xy_robot_base_m.value)
            if other is not None:
                if predicate in (VisualPredicate.LEFT_OF, VisualPredicate.RIGHT_OF, VisualPredicate.IN_FRONT_OF):
                    sx0, sx1, sy0, sy1 = _bounds(footprint)
                    rx0, rx1, ry0, ry1 = _bounds(other)
                    gap = (sy0 - ry1 if predicate is VisualPredicate.LEFT_OF else
                        ry0 - sy1 if predicate is VisualPredicate.RIGHT_OF else sx0 - rx1)
                    required = parameters.get("visual_relation_config.directional_gap_m")
                    values["placement_directional_gap_m"] = gap
                    if required is not None:
                        values["placement_directional_gap_deficit_m"] = max(0.0, required - gap)
                elif predicate in (VisualPredicate.NEAR, VisualPredicate.SEPARATED):
                    clearance = polygon_clearance_v1(footprint, other)
                    name = ("near_maximum_clearance_m" if predicate is VisualPredicate.NEAR
                        else "separated_minimum_clearance_m")
                    required = parameters.get("visual_relation_config." + name)
                    values["placement_footprint_clearance_m"] = clearance
                    if required is not None:
                        values["placement_clearance_deficit_m"] = max(0.0,
                            clearance - required if predicate is VisualPredicate.NEAR else required - clearance)
                elif predicate is VisualPredicate.ON_SURFACE:
                    values["placement_support_approach_clearance_m"] = polygon_clearance_v1(footprint, other)
                    fraction = _xy_overlap_fraction(footprint, other)
                    required = parameters.get("visual_relation_config.support_minimum_overlap_fraction")
                    values["placement_overlap_fraction"] = fraction
                    if required is not None:
                        values["placement_overlap_fraction_deficit"] = max(0.0, required - fraction)
                        values["placement_overlap_fraction_margin"] = fraction - required
    source = "bounded_command_geometry" if predicted else "rgbd_geometry"
    return tuple(PhaseMeasurementV1(name, values.get(name), unit, source) for name, unit in names)


@dataclass(frozen=True, slots=True)
class PickXYErrorV1:
    delta_x_m: float
    delta_y_m: float
    distance_m: float


def pick_xy_error_v1(snapshot: StateSnapshotV2, subject_ref: str) -> PickXYErrorV1 | None:
    subject = next(
        (entity for entity in snapshot.visual_scene.entities if entity.track_id == subject_ref),
        None,
    )
    top = (None if subject is None or subject.visibility is not Visibility.CLEAR
           else subject.geometry.top_center_robot_base_m.value)
    if top is None:
        return None
    tcp = snapshot.robot.tcp_position_robot_base_m
    delta_x = top[0] - tcp[0]
    delta_y = top[1] - tcp[1]
    return PickXYErrorV1(delta_x, delta_y, hypot(delta_x, delta_y))


def projected_pick_xy_distance_v1(
    error: PickXYErrorV1,
    action: AtomicAction,
    translation_step_m: float,
) -> float | None:
    if not isfinite(translation_step_m) or translation_step_m <= 0.0:
        raise ValueError("translation_step_m must be finite and positive")
    if action is AtomicAction.HOLD:
        return error.distance_m
    direction = TRANSLATION_DIRECTIONS_ROBOT_BASE.get(action)
    if direction is None or direction[2] != 0.0:
        return None
    return hypot(
        error.delta_x_m - direction[0] * translation_step_m,
        error.delta_y_m - direction[1] * translation_step_m,
    )


def point_xy_error_v1(snapshot: StateSnapshotV2, reference_ref: str | None) -> PickXYErrorV1 | None:
    region = next((item for item in snapshot.visual_scene.regions if item.region_id == reference_ref), None)
    point = None if region is None or region.geometry.point_robot_base_m is None else region.geometry.point_robot_base_m.value
    if point is None:
        return None
    tcp = snapshot.robot.tcp_position_robot_base_m
    dx, dy = point[0] - tcp[0], point[1] - tcp[1]
    return PickXYErrorV1(dx, dy, hypot(dx, dy))


def has_xy_alignment_objective_v1(snapshot: StateSnapshotV2) -> bool:
    context = snapshot.local_intent
    if (context is not None and context.execution is not None
        and context.execution.status in ("active", "completed")
        and context.execution.candidate.metric_kind == "tcp_xy"):
        return True
    return (snapshot.skill.skill_kind, snapshot.skill.phase) in (
        ("pick", "align_subject_xy"), ("press", "align_press_point"))


def active_xy_alignment_error_v1(snapshot: StateSnapshotV2) -> PickXYErrorV1 | None:
    context = snapshot.local_intent
    if (context is not None and context.execution is not None
        and context.execution.status in ("active", "completed")
        and context.execution.candidate.metric_kind == "tcp_xy"):
        target = context.execution.candidate.target_robot_base_m
        tcp = snapshot.robot.tcp_position_robot_base_m
        dx, dy = target[0] - tcp[0], target[1] - tcp[1]
        return PickXYErrorV1(dx, dy, hypot(dx, dy))
    if snapshot.skill.skill_kind == "pick" and snapshot.skill.phase == "align_subject_xy":
        return pick_xy_error_v1(snapshot, snapshot.skill.subject_ref)
    if snapshot.skill.skill_kind == "press" and snapshot.skill.phase == "align_press_point":
        return point_xy_error_v1(snapshot, snapshot.skill.target_ref)
    return None
