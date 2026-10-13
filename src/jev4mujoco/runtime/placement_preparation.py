"""区域放置的独立水平准备；只构造未排序选项和核对公开效果。"""
from dataclasses import asdict, replace
from math import hypot, isfinite

from jev4mujoco.contracts.actions import AtomicAction, TRANSLATION_DIRECTIONS_ROBOT_BASE
from jev4mujoco.contracts.local_intent import LocalIntentCandidateV1, PlacementMarginPreparationV1
from jev4mujoco.contracts.visual_scene import Visibility, VisualPredicate
from jev4mujoco.runtime.progress import placement_horizontal_measurements_v1
from jev4mujoco.perception.surface_geometry import polygon_contains_polygon_v1


def _current(snapshot):
    # 延迟导入以复用同一份已绑定的当前证据与权限判断。
    from jev4mujoco.runtime.local_intent import (
        current_runtime_measurements_v1, placement_horizontal_preparation_allowed_v1,
    )
    skill = snapshot.skill
    if (skill.goal is None or skill.goal.predicate not in (VisualPredicate.IN_REGION, VisualPredicate.INSIDE)
        or not skill.goal.desired_value or not placement_horizontal_preparation_allowed_v1(snapshot)):
        return None
    entities = {e.track_id:e for e in snapshot.visual_scene.entities}
    subject = entities.get(skill.subject_ref)
    region = next((r for r in snapshot.visual_scene.regions if r.region_id==skill.target_ref), None)
    if region is None or subject is None:
        return None
    refs = [subject]
    if region.owner_entity_ref is not None:
        refs.append(entities.get(region.owner_entity_ref))
    if any(e is None or e.visibility is not Visibility.CLEAR or any(v.value is None for v in (
        e.geometry.centroid_robot_base_m, e.geometry.extent_m, e.geometry.top_center_robot_base_m,
        e.geometry.footprint_xy_robot_base_m)) for e in refs):
        return None
    boundary = region.geometry.boundary_xy_robot_base_m
    values = current_runtime_measurements_v1(snapshot)
    margin = values.get("placement_containment_margin_m")
    measurement = next((m for m in snapshot.phase_feedback.measurements if m.name=="placement_containment_margin_m"), None)
    if (boundary is None or boundary.value is None or margin is None or margin < 0
        or measurement is None or measurement.unit!="m" or measurement.source!="rgbd_geometry"):
        return None
    return subject, region, margin


def placement_preparation_comparable_v1(candidate, snapshot):
    evidence = candidate.placement_preparation
    current = _current(snapshot)
    if evidence is None or current is None:
        return False
    skill, scene = snapshot.skill, snapshot.visual_scene
    return (skill.skill_kind=="place" and skill.phase=="lower_to_release"
        and (evidence.task_id, evidence.skill_instance_id, evidence.subject_ref,
            evidence.source_ref, evidence.target_ref, evidence.goal)
        == (snapshot.task.task_id, skill.skill_instance_id, skill.subject_ref,
            skill.source_ref, skill.target_ref, (skill.goal.predicate.value, skill.goal.reference_ref, skill.goal.desired_value))
        and (evidence.scene_id, evidence.sensor_id, evidence.calibration_id)
        == (scene.scene_id, scene.sensor.sensor_id, scene.sensor.calibration_id)
        and evidence.region_boundary_xy_m==current[1].geometry.boundary_xy_robot_base_m.value
        and evidence.baseline_state_id<=snapshot.state_id
        and evidence.baseline_action_epoch<=snapshot.action_epoch
        and evidence.baseline_observation_id<=scene.observation_id
        and evidence.baseline_capture_timestamp_s<=scene.capture_timestamp_s)


def placement_preparation_candidates_v1(snapshot, *, step_m=None, tolerance_m=None,
                                        minimum_improvement_m=None, retained=None):
    current = _current(snapshot)
    if current is None:
        return ()
    actions = tuple(a for a in snapshot.skill.allowed_actions if a in (
        "forward", "backward", "left", "right", "hold"))
    if not actions or not any(a!="hold" for a in actions):
        return ()
    if retained is not None:
        return ((replace(retained, allowed_actions=actions),)
            if placement_preparation_comparable_v1(retained, snapshot) else ())
    if any(v is None or not isfinite(v) or v<=0 for v in (step_m, tolerance_m, minimum_improvement_m)):
        return ()  # 没有部署参数就不补造默认步长或容差。
    subject, region, margin = current
    skill, scene, tcp = snapshot.skill, snapshot.visual_scene, snapshot.robot.tcp_position_robot_base_m
    candidates = []
    for action in ("forward", "backward", "left", "right"):
        if action not in actions:
            continue
        direction = TRANSLATION_DIRECTIONS_ROBOT_BASE[AtomicAction(action)]
        delta = tuple(step_m*d for d in direction)
        footprint = tuple((x+delta[0], y+delta[1]) for x,y in subject.geometry.footprint_xy_robot_base_m.value)
        if not polygon_contains_polygon_v1(region.geometry.boundary_xy_robot_base_m.value, footprint):
            continue
        projected = {m.name:m.value for m in placement_horizontal_measurements_v1(
            snapshot, dict(snapshot.phase_feedback.parameters), evidence_current=True,
            delta_xy=delta[:2], predicted=True)}.get("placement_containment_margin_m")
        if projected is None or projected-margin+1e-12<minimum_improvement_m:
            continue
        evidence = PlacementMarginPreparationV1(snapshot.task.task_id, skill.skill_instance_id,
            skill.subject_ref, skill.source_ref, skill.target_ref,
            (skill.goal.predicate.value, skill.goal.reference_ref, skill.goal.desired_value),
            snapshot.state_id, snapshot.action_epoch, scene.observation_id, scene.capture_timestamp_s,
            scene.scene_id, scene.sensor.sensor_id, scene.sensor.calibration_id,
            region.geometry.boundary_xy_robot_base_m.value, tcp, margin, projected, minimum_improvement_m)
        candidates.append(LocalIntentCandidateV1("prepare_placement_margin_"+action,
            "Improve the bound placement containment margin at the existing verified clearance. Reach this selected XY preparation point AND observe an actual native margin improvement from its original frame. This does not complete descent, release or certify the future path.",
            actions, "tcp_xy", tuple(x+d for x,d in zip(tcp,delta)), tolerance_m, skill.target_ref,
            ("placement_relation_until_release", "avoid_other_objects"),
            geometry_source="rgbd_robot_margin_preparation", placement_preparation=evidence))
    return tuple(candidates)


def placement_preparation_goal_reached_v1(candidate, snapshot):
    if not placement_preparation_comparable_v1(candidate, snapshot):
        return False
    p = candidate.placement_preparation
    scene = snapshot.visual_scene
    return (snapshot.state_id>p.baseline_state_id and snapshot.action_epoch>p.baseline_action_epoch
        and scene.observation_id>p.baseline_observation_id and scene.capture_timestamp_s>p.baseline_capture_timestamp_s
        and hypot(*(candidate.target_robot_base_m[i]-snapshot.robot.tcp_position_robot_base_m[i]
                    for i in (0,1)))<=candidate.tolerance_m
        and _current(snapshot)[2]-p.baseline_margin_m+1e-12>=p.minimum_improvement_m)


def placement_preparation_context_v1(candidate, snapshot):
    p = candidate.placement_preparation
    comparable = placement_preparation_comparable_v1(candidate, snapshot)
    margin = _current(snapshot)[2] if comparable else None
    observed = (comparable and snapshot.state_id>p.baseline_state_id
        and snapshot.action_epoch>p.baseline_action_epoch
        and snapshot.visual_scene.observation_id>p.baseline_observation_id
        and snapshot.visual_scene.capture_timestamp_s>p.baseline_capture_timestamp_s)
    baseline = asdict(p)
    baseline.pop("region_boundary_xy_m")  # 运行时保存完整绑定；模型已有当前目标区域，不重复多边形。
    return {"choice_id":candidate.choice_id, "baseline":baseline,
        "target_robot_base_m":candidate.target_robot_base_m,
        "tcp_tolerance_m":candidate.tolerance_m, "current_evidence_comparable":comparable,
        "current_margin_m":margin,
        "predicted_margin_improvement_m":p.predicted_margin_m-p.baseline_margin_m,
        "observed_margin_improvement_m":margin-p.baseline_margin_m if observed else None,
        "local_goal_reached":placement_preparation_goal_reached_v1(candidate,snapshot),
        "note":"Baseline and prediction retain their original capture identity; rigid-follow geometry is not executed motion, safe descent or final success. Unknown current evidence cannot certify the preparation."}
