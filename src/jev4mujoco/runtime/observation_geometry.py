"""释放后的任务观测目标；历史只给身份范围，不给当前物体几何。"""
from dataclasses import dataclass

from jev4mujoco.contracts.visual_scene import EntityVisibilityDiagnosticV1, Visibility


def required_task_observation_refs_v1(snapshot):
    refs = [snapshot.skill.subject_ref, *snapshot.task.completed_subject_refs]
    target = snapshot.skill.target_ref
    if any(entity.track_id == target for entity in snapshot.visual_scene.entities):
        refs.append(target)
    region = next((r for r in snapshot.visual_scene.regions if r.region_id == target), None)
    if region is not None and region.owner_entity_ref is not None:
        refs.append(region.owner_entity_ref)
    return tuple(dict.fromkeys(ref for ref in refs if ref is not None))


def current_visibility_diagnostic_v1(entity, scene):
    diagnostic = entity.measurement.visibility_diagnostic
    if (not isinstance(diagnostic, EntityVisibilityDiagnosticV1)
        or (diagnostic.scene_id, diagnostic.observation_id, diagnostic.capture_timestamp_s)
        != (scene.scene_id, scene.observation_id, scene.capture_timestamp_s)):
        return None
    return diagnostic


@dataclass(frozen=True, slots=True)
class TaskObservationGeometryV1:
    required_refs: tuple[str, ...]
    geometry_observed: bool | None
    span_deficit_m: float | None
    robot_occlusion_hint: bool | None


def task_observation_geometry_v1(snapshot):
    refs = required_task_observation_refs_v1(snapshot)
    scene = snapshot.visual_scene
    action_end = 0.0 if snapshot.recent_action is None else snapshot.recent_action.end_timestamp_s
    if (not snapshot.freshness.visual_is_fresh
        or snapshot.sources.visual_observation_id != scene.observation_id
        or snapshot.sources.visual_capture_timestamp_s != scene.capture_timestamp_s
        or scene.capture_timestamp_s < action_end):
        return TaskObservationGeometryV1(refs, None, None, None)
    entities = {e.track_id: e for e in scene.entities}
    observed, errors, hints = [], [], []
    for ref in refs:
        entity = entities.get(ref)
        if entity is None:
            observed.append(None)
            errors.append(None)
            hints.append(None)
            continue
        geometry = entity.geometry
        complete = entity.visibility is Visibility.CLEAR and all(value.value is not None for value in (
            geometry.centroid_robot_base_m, geometry.top_center_robot_base_m,
            geometry.extent_m, geometry.footprint_xy_robot_base_m))
        observed.append(bool(complete))
        if complete:
            errors.append(0.0)
            continue
        diagnostic = current_visibility_diagnostic_v1(entity, scene)
        usable = (entity.visibility is Visibility.PARTIAL and diagnostic is not None
            and diagnostic.valid_point_count >= 4 and diagnostic.visible_xy_span_m is not None
            and diagnostic.reference_scene_id == scene.scene_id
            and diagnostic.reference_minimum_span_m is not None
            and diagnostic.reference_observation_id is not None and diagnostic.span_deficit_m is not None
            and diagnostic.span_deficit_m > 0
            and diagnostic.reason == "visible_span_below_reference")
        errors.append(float(diagnostic.span_deficit_m) if usable else None)
        hints.append((diagnostic.robot_adjacent_pixel_count > 0) if usable and
            diagnostic.robot_mask_source == "mujoco_ideal_robot_segmentation" else None)
    return TaskObservationGeometryV1(refs,
        None if not observed or any(value is None for value in observed) else all(observed),
        None if any(error is None for error in errors) or not errors else max(errors),
        True if any(hint is True for hint in hints) else None if any(hint is None for hint in hints) else False)
