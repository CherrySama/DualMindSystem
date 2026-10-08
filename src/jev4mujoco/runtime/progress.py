from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from jev4mujoco.contracts.actions import AtomicAction, TRANSLATION_DIRECTIONS_ROBOT_BASE
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2
from jev4mujoco.contracts.visual_scene import Visibility


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
