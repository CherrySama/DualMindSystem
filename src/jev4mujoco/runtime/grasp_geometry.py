from __future__ import annotations

from jev4mujoco.contracts.actions import AtomicAction
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2
from jev4mujoco.contracts.visual_scene import Visibility


def grasp_height_band_from_geometry_v1(
    centroid: tuple[float, float, float], extent: tuple[float, float, float]
) -> tuple[float, float]:
    half_band = 0.10 * extent[2]
    return centroid[2] - half_band, centroid[2] + half_band


def grasp_height_band_v1(
    snapshot: StateSnapshotV2, subject_ref: str
) -> tuple[float, float] | None:
    """Keep the TCP in the central fifth of the observed object's height."""
    subject = next(
        (entity for entity in snapshot.visual_scene.entities if entity.track_id == subject_ref),
        None,
    )
    if subject is None:
        return None
    centroid = subject.geometry.centroid_robot_base_m.value
    extent = subject.geometry.extent_m.value
    if centroid is None or extent is None:
        return None
    return grasp_height_band_from_geometry_v1(centroid, extent)


def grasp_translation_step_v1(
    snapshot: StateSnapshotV2, subject_ref: str, action: AtomicAction,
    maximum_step_m: float,
) -> float:
    """Bound a grasp-alignment step by the visual error on its action axis."""
    band = grasp_height_band_v1(snapshot, subject_ref)
    if band is None:
        return maximum_step_m
    subject = next(
        entity for entity in snapshot.visual_scene.entities if entity.track_id == subject_ref
    )
    centroid = subject.geometry.centroid_robot_base_m.value
    if centroid is None:
        return maximum_step_m
    axis = (
        0 if action in (AtomicAction.FORWARD, AtomicAction.BACKWARD)
        else 1 if action in (AtomicAction.LEFT, AtomicAction.RIGHT)
        else 2
    )
    target = 0.5 * (band[0] + band[1]) if axis == 2 else centroid[axis]
    remaining = abs(target - snapshot.robot.tcp_position_robot_base_m[axis])
    return min(maximum_step_m, max(0.001, remaining))


def grasp_pose_correction_v1(
    snapshot: StateSnapshotV2, subject_ref: str,
) -> tuple[float, float, float] | None:
    """Signed correction to centroid XY and the nearest grasp-height-band edge."""
    subject = next((item for item in snapshot.visual_scene.entities if item.track_id == subject_ref), None)
    if subject is None or subject.visibility is not Visibility.CLEAR:
        return None
    centroid = subject.geometry.centroid_robot_base_m.value
    band = grasp_height_band_v1(snapshot, subject_ref)
    if centroid is None or band is None:
        return None
    tcp = snapshot.robot.tcp_position_robot_base_m
    return (centroid[0] - tcp[0], centroid[1] - tcp[1],
            max(band[0] - tcp[2], 0.0) - max(tcp[2] - band[1], 0.0))
