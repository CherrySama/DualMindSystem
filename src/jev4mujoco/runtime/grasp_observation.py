"""抓取下降的观测回撤和模型选择的位姿准备；不读取隐藏物体/接触状态。"""
from dataclasses import asdict, replace
from math import hypot, sqrt

from jev4mujoco.contracts.actions import AtomicAction, TRANSLATION_DIRECTIONS_ROBOT_BASE
from jev4mujoco.contracts.grasp_approach import GraspApproachV1, GraspPreparationV1, GraspViewObservationV1
from jev4mujoco.contracts.state_snapshot import GraspState
from jev4mujoco.contracts.visual_scene import Visibility
from jev4mujoco.runtime.grasp_geometry import grasp_pose_correction_v1


VIEW_OFFSETS = (("positive_x", 1., 0.), ("negative_x", -1., 0.),
                ("positive_y", 0., 1.), ("negative_y", 0., -1.))


def _entity(snapshot):
    return next((e for e in snapshot.visual_scene.entities if e.track_id == snapshot.skill.subject_ref), None)


def _fresh(snapshot, maximum_age=.5):
    return (snapshot.freshness.visual_is_fresh and snapshot.freshness.visual_age_s <= maximum_age
        and snapshot.sources.robot_measurement_timestamp_s == snapshot.timestamp_s
        and snapshot.visual_scene.capture_timestamp_s <= snapshot.timestamp_s
        and (snapshot.recent_action is None or snapshot.visual_scene.capture_timestamp_s >= snapshot.recent_action.end_timestamp_s))


def _diagnostic(snapshot):
    entity = _entity(snapshot)
    d = None if entity is None else entity.measurement.visibility_diagnostic
    if (d is None or (d.scene_id, d.observation_id, d.capture_timestamp_s) != (
        snapshot.visual_scene.scene_id, snapshot.visual_scene.observation_id, snapshot.visual_scene.capture_timestamp_s)
        or d.reference_scene_id != snapshot.visual_scene.scene_id or d.reference_observation_id is None
        or d.reference_observation_id > d.observation_id or d.reference_capture_timestamp_s > d.capture_timestamp_s):
        return None
    return d


def _sample(snapshot):
    entity = _entity(snapshot)
    if entity is None:
        return None
    d = _diagnostic(snapshot)
    complete = entity.visibility is Visibility.CLEAR
    scene = snapshot.visual_scene
    return GraspViewObservationV1(snapshot.state_id, snapshot.action_epoch, scene.observation_id,
        scene.capture_timestamp_s, scene.scene_id, scene.sensor.sensor_id, scene.sensor.calibration_id,
        snapshot.robot.tcp_position_robot_base_m,
        entity.geometry.centroid_robot_base_m.value if complete else None,
        entity.geometry.extent_m.value if complete else None,
        None if d is None else d.span_deficit_m, entity.visibility.value)


def current_grasp_approach_v1(snapshot):
    r = snapshot.grasp_approach
    if r is None or (r.task_id, r.skill_instance_id, r.subject_ref, r.source_ref) != (
        snapshot.task.task_id, snapshot.skill.skill_instance_id, snapshot.skill.subject_ref, snapshot.skill.source_ref):
        return None
    c = r.current
    if (snapshot.skill.skill_kind != "pick" or snapshot.skill.phase != "descend_to_grasp"
        or (c.state_id, c.action_epoch, c.observation_id, c.capture_timestamp_s, c.scene_id, c.sensor_id, c.calibration_id)
        != (snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id,
            snapshot.visual_scene.capture_timestamp_s, snapshot.visual_scene.scene_id,
            snapshot.visual_scene.sensor.sensor_id, snapshot.visual_scene.sensor.calibration_id)):
        return None
    return r


def grasp_geometry_observed_v1(snapshot):
    entity = _entity(snapshot)
    return (_fresh(snapshot) and entity is not None and entity.visibility is Visibility.CLEAR
        and entity.geometry.centroid_robot_base_m.value is not None and entity.geometry.extent_m.value is not None)


def grasp_empty_tool_ready_v1(snapshot, opening=.04, tolerance=.002):
    g = snapshot.robot.gripper
    return (_fresh(snapshot) and snapshot.physical.grasp_state is GraspState.NOT_HELD
        and abs(g.opening_m-opening) <= tolerance and abs(g.target_opening_m-opening) <= tolerance
        and snapshot.robot.tool_geometry is not None)


def grasp_observation_motion_ready_v1(snapshot, opening=.04, tolerance=.002):
    r, d = current_grasp_approach_v1(snapshot), _diagnostic(snapshot)
    source = next((x for x in snapshot.visual_scene.regions if x.region_id == snapshot.skill.source_ref), None)
    height = None if source is None or source.geometry.surface_height_robot_base_m is None else source.geometry.surface_height_robot_base_m.value
    tool = snapshot.robot.tool_geometry
    return bool(r is not None and r.geometry_comparable and r.blocked_reason is None
        and grasp_empty_tool_ready_v1(snapshot, opening, tolerance)
        and d is not None and d.robot_adjacent_pixel_count > 0 and d.valid_point_count >= 4
        and d.span_deficit_m is not None and d.span_deficit_m > 0
        and height is not None and tool is not None
        and min(v[2] for part in tool.parts for v in part.vertices_robot_base_m) >= height)


def grasp_preparation_options_v1(snapshot, xy_tolerance, preparation_tolerance):
    r = current_grasp_approach_v1(snapshot)
    if r is None or not r.geometry_comparable or not grasp_geometry_observed_v1(snapshot):
        return ()
    center = _entity(snapshot).geometry.centroid_robot_base_m.value
    z = snapshot.robot.tcp_position_robot_base_m[2]
    # 原XY可接受圆内的四个几何目标；既不排序，也不预测未来可见性。
    return tuple(("prepare_grasp_view_"+name, (center[0]+dx*xy_tolerance, center[1]+dy*xy_tolerance, z),
        (dx*xy_tolerance, dy*xy_tolerance), preparation_tolerance) for name,dx,dy in VIEW_OFFSETS)


def selected_grasp_target_v1(snapshot):
    r = current_grasp_approach_v1(snapshot)
    if r is None or r.preparation is None or not r.geometry_comparable or not grasp_geometry_observed_v1(snapshot):
        return None
    p, center = r.preparation, _entity(snapshot).geometry.centroid_robot_base_m.value
    return (center[0]+p.offset_xy_m[0], center[1]+p.offset_xy_m[1], snapshot.robot.tcp_position_robot_base_m[2])


def grasp_approach_context_v1(snapshot, *, include_options=True):
    """把当前/历史位姿的数值关系显式投影，不预测遮挡或替模型选择。"""
    r=current_grasp_approach_v1(snapshot)
    if r is None:
        return {"current_evidence_matches": False}
    tcp=snapshot.robot.tcp_position_robot_base_m
    def distances(point):
        return [{"observation_id": past.observation_id, "action_epoch": past.action_epoch,
            "xy_distance_m": hypot(point[0]-past.tcp_robot_base_m[0],point[1]-past.tcp_robot_base_m[1])}
            for past in r.failures]
    result={**asdict(r), "current_evidence_matches": True,
        "current_xy_distance_to_occluded_poses": distances(tcp),
        "future_visibility": None,
        "height_handoff_ready": grasp_height_handoff_ready_v1(snapshot),
        "height_goal_role": "restore_after_model_selected_preparation" if r.stage == "prepare"
            else "deferred_until_fresh_observation_and_selected_preparation" if r.stage in ("observe","unknown")
            else "original_grasp_condition",
        "preparation_role": "change_approach_xy_after_observation_loss; local completion never proves grasp",
        "progress_scopes": {
            "actions_without_new_best": "original grasp pose; paused during observation/preparation",
            "observation_nonprogress_actions": "current observation obligation; fresh span/full geometry only",
            "preparation_nonprogress_actions": "actual model-selected preparation; fresh XY error only",
            "replan_count": "Runtime local intent budget across all steps; never reset by restoring visibility"}}
    if include_options:
        result["preparation_options"]=[{"choice_id": name,"target_robot_base_m": target,
            "current_xy_distance_m": hypot(target[0]-tcp[0],target[1]-tcp[1]),
            "target_xy_distance_to_occluded_poses": distances(target),"future_visibility": None}
            for name,target,_,_ in grasp_preparation_options_v1(snapshot,r.xy_tolerance_m,r.preparation_tolerance_m)]
    return result


def grasp_recovery_mode_v1(snapshot):
    if snapshot.skill.skill_kind != "pick" or snapshot.skill.phase != "descend_to_grasp" or snapshot.grasp_approach is None:
        return None
    r = current_grasp_approach_v1(snapshot)
    if r is None or not r.geometry_comparable or r.blocked_reason is not None or not _fresh(snapshot):
        return "unknown"
    if r.stage == "approach" and not grasp_height_handoff_ready_v1(snapshot):
        return "prepare"
    return r.stage


def grasp_preparation_ready_v1(snapshot):
    support = snapshot.physical.support_state
    r = current_grasp_approach_v1(snapshot)
    return (r is not None and r.geometry_comparable and r.blocked_reason is None and grasp_geometry_observed_v1(snapshot)
        and grasp_empty_tool_ready_v1(snapshot) and support.subject_ref == snapshot.skill.subject_ref
        and support.supported is True)


def grasp_preparation_goal_reached_v1(candidate, snapshot):
    if not grasp_preparation_ready_v1(snapshot):
        return False
    r = current_grasp_approach_v1(snapshot)
    center, tcp = r.current.centroid_robot_base_m, snapshot.robot.tcp_position_robot_base_m
    target = candidate.target_robot_base_m
    return (hypot(target[0]-tcp[0], target[1]-tcp[1]) <= candidate.tolerance_m
        and hypot(center[0]-tcp[0], center[1]-tcp[1]) <= r.xy_tolerance_m)


def grasp_height_handoff_ready_v1(snapshot):
    r = current_grasp_approach_v1(snapshot)
    if r is None or r.preparation is None or not grasp_preparation_ready_v1(snapshot):
        return False
    target, tcp = selected_grasp_target_v1(snapshot), snapshot.robot.tcp_position_robot_base_m
    center = r.current.centroid_robot_base_m
    return (target is not None and hypot(target[0]-tcp[0], target[1]-tcp[1]) <= r.preparation.tolerance_m
        and hypot(center[0]-tcp[0], center[1]-tcp[1]) <= r.xy_tolerance_m)


def grasp_recovery_candidates_v1(snapshot, *, for_validation=False):
    from jev4mujoco.contracts.local_intent import LocalIntentCandidateV1, LocalIntentConditionV1
    r, mode = current_grasp_approach_v1(snapshot), grasp_recovery_mode_v1(snapshot)
    if r is None:
        return ()
    candidates=[]
    if mode in ("observe", "unknown") or for_validation:
        actions=("up","hold") if grasp_observation_motion_ready_v1(snapshot) else ("hold",)
        if not for_validation:
            actions=tuple(x for x in actions if x in snapshot.skill.allowed_actions)
        candidates.append(LocalIntentCandidateV1("observe_grasp_geometry",
            "Restore current complete grasp geometry by retracting the open unheld tool along the measured approach; current support remains unknown until observed.",
            actions,"runtime_conditions",reference_ref=snapshot.skill.subject_ref,
            maintain=("gripper_open","subject_not_held"),geometry_source="fresh_fragment_and_historical_robot_approach",
            completion_conditions=(LocalIntentConditionV1("grasp_geometry_observed","equals",True),)))
    if mode in ("prepare","approach") and grasp_preparation_ready_v1(snapshot):
        for name,target,_,tolerance in grasp_preparation_options_v1(snapshot,r.xy_tolerance_m,r.preparation_tolerance_m):
            if (r.preparation is None or r.preparation.choice_id == name or for_validation):
                actions=tuple(x for x in ("forward","backward","left","right","hold")
                    if for_validation or x in snapshot.skill.allowed_actions)
                candidates.append(LocalIntentCandidateV1(name,
                    "Prepare an alternative XY approach within the original grasp tolerance, preserving height; compare prior occluded poses. Future visibility is unknown, not guaranteed. Original grasp conditions still govern closing.",
                    actions,"tcp_xy",target,tolerance,
                    snapshot.skill.subject_ref,("subject_supported","gripper_open"),"current_rgbd_grasp_tolerance_region"))
    return tuple(candidates)


def grasp_approach_translation_step_v1(snapshot, action, maximum_step, preparation_target=None):
    """执行/JEV投影共用；只限制机器人目标，不预测视觉或物体运动。"""
    r, mode=current_grasp_approach_v1(snapshot),grasp_recovery_mode_v1(snapshot)
    if r is None:
        return None
    tcp=snapshot.robot.tcp_position_robot_base_m
    if mode == "observe" and action is AtomicAction.UP:
        remaining=r.return_observation.tcp_robot_base_m[2]+r.retraction_margin_m-tcp[2]
        return min(maximum_step,max(.001,remaining))
    direction=TRANSLATION_DIRECTIONS_ROBOT_BASE.get(action)
    if mode not in ("prepare","approach") or direction is None or direction[2] != 0:
        return None
    center=r.current.centroid_robot_base_m
    if center is None:
        return None
    target=preparation_target or selected_grasp_target_v1(snapshot)
    if target is None:
        # 一层模式只选原子动作，不由Runtime替它选一个准备位姿。
        if snapshot.local_intent is not None:
            return None
        step=maximum_step
    else:
        axis=0 if direction[0] else 1
        step=min(maximum_step,max(.001,abs(target[axis]-tcp[axis])))
    dx,dy=tcp[0]-center[0],tcp[1]-center[1]
    if hypot(dx,dy) <= r.xy_tolerance_m:
        along=dx*direction[0]+dy*direction[1]
        limit=-along+sqrt(max(0.,along*along+r.xy_tolerance_m**2-dx*dx-dy*dy))
        step=min(step,max(0.,limit))
    return step


class GraspObservationRuntimeV1:
    def __init__(self, config, xy_tolerance, translation_error, max_visual_age, opening, gripper_tolerance):
        self.config, self.xy_tolerance = config, xy_tolerance
        self.translation_error, self.preparation_tolerance = translation_error, sqrt(2)*translation_error
        self.max_age, self.opening, self.gripper_tolerance = max_visual_age, opening, gripper_tolerance
        self.reset()

    def reset(self):
        self._key = self._previous = self._record = None

    def attach(self, snapshot, action=None):
        skill = snapshot.skill
        if skill.skill_kind != "pick" or skill.phase != "descend_to_grasp":
            self.reset()
            return snapshot
        key = (snapshot.task.task_id, skill.skill_instance_id, skill.subject_ref, skill.source_ref)
        if key != self._key:
            self.reset(); self._key = key
        if not _fresh(snapshot, self.max_age):
            return replace(snapshot, grasp_approach=self._record)
        current, previous = _sample(snapshot), self._previous
        if current is None or (previous is not None and (current.observation_id < previous.observation_id
            or current.capture_timestamp_s < previous.capture_timestamp_s)):
            return replace(snapshot, grasp_approach=self._record)
        d = _diagnostic(snapshot)
        lost_after_down = (current.visibility == "partial" and d is not None
            and d.span_deficit_m is not None and d.span_deficit_m > 0 and d.robot_adjacent_pixel_count > 0
            and previous is not None and previous.visibility == "clear"
            and previous.centroid_robot_base_m is not None and previous.extent_m is not None
            and action is not None and action.executed and action.action is AtomicAction.DOWN
            and action.based_on_state_id == previous.state_id
            and (action.start_action_epoch, action.end_action_epoch) == (previous.action_epoch, current.action_epoch)
            and current.observation_id > previous.observation_id
            and previous.capture_timestamp_s <= action.start_simulation_time_s
            and current.capture_timestamp_s >= action.end_simulation_time_s
            and (current.scene_id, current.sensor_id, current.calibration_id) == (
                previous.scene_id, previous.sensor_id, previous.calibration_id)
            and grasp_empty_tool_ready_v1(snapshot, self.opening, self.gripper_tolerance))
        before = r = self._record
        if lost_after_down:
            reason = "fresh grasp geometry lost after executed descent; model-selected preparation required after observation recovery"
            if r is None:
                r = GraspApproachV1(*key, current, previous, (current,), (), "observe", reason,
                    self.xy_tolerance, self.translation_error, self.preparation_tolerance,
                    initial_observation=previous, last_counted_action_epoch=previous.action_epoch,
                    best_observation_deficit_m=current.span_deficit_m)
            else:
                failed = () if r.preparation is None else (r.preparation,)
                r = replace(r, current=current, return_observation=previous, failures=r.failures+(current,),
                    failed_preparations=r.failed_preparations+tuple(p.choice_id for p in failed),
                    failed_preparation_records=r.failed_preparation_records+failed,
                    stage="observe", reason=reason, preparation=None,
                    best_observation_deficit_m=current.span_deficit_m, best_preparation_error_m=None)
            if r.best_observed_height_error_m is None:
                half=.1*previous.extent_m[2]
                best=max(0., abs(previous.tcp_robot_base_m[2]-previous.centroid_robot_base_m[2])-half)
                r=replace(r,best_observed_height_error_m=best)
        if r is not None:
            origin = r.initial_observation or r.return_observation
            comparable = r.geometry_comparable and (origin.scene_id, origin.sensor_id, origin.calibration_id) == (
                current.scene_id, current.sensor_id, current.calibration_id)
            if current.centroid_robot_base_m is not None:
                comparable = comparable and sqrt(sum((a-b)**2 for a,b in zip(
                    origin.centroid_robot_base_m,current.centroid_robot_base_m))) <= self.config.maximum_reference_shift_m
            p=r.preparation
            if p is not None and current.centroid_robot_base_m is not None:
                comparable = comparable and sqrt(sum((a-b)**2 for a,b in zip(
                    p.origin.centroid_robot_base_m,current.centroid_robot_base_m))) <= self.config.maximum_reference_shift_m
            stage = "unknown" if not comparable else "observe" if not grasp_geometry_observed_v1(snapshot) else "prepare"
            if comparable and current.centroid_robot_base_m is not None and p is not None:
                target=(current.centroid_robot_base_m[0]+p.offset_xy_m[0],current.centroid_robot_base_m[1]+p.offset_xy_m[1])
                if hypot(*(a-b for a,b in zip(target,current.tcp_robot_base_m))) <= p.tolerance_m:
                    stage="approach"
            r=replace(r,current=current,stage=stage,geometry_comparable=comparable)
            bound=replace(snapshot,grasp_approach=r)
            if stage == "approach" and not grasp_height_handoff_ready_v1(bound):
                r=replace(r,stage="prepare")
                bound=replace(bound,grasp_approach=r)
            r=self._observe_progress(r,before,bound,action,previous)
            self._record=r
        if previous is None or current.observation_id >= previous.observation_id:
            self._previous=current
        return replace(snapshot,grasp_approach=r)

    def _observe_progress(self, record, before, snapshot, action, previous):
        current=record.current
        if (action is None or not action.executed or previous is None
            or current.action_epoch <= record.last_counted_action_epoch
            or action.based_on_state_id != previous.state_id
            or (action.start_action_epoch,action.end_action_epoch) != (previous.action_epoch,current.action_epoch)
            or current.observation_id <= previous.observation_id
            or previous.capture_timestamp_s > action.start_simulation_time_s
            or current.capture_timestamp_s < action.end_simulation_time_s):
            return record
        # 归属使用动作前的义务；下降后的PARTIAL不能把下降尝试改算为观察。
        scope=("grasp_pose" if before is None or action.action is AtomicAction.DOWN
            else "observation" if before.stage == "observe"
            else "preparation" if before.stage == "prepare"
            else "grasp_pose" if before.stage == "approach" else "unknown")
        changes={"last_counted_action_epoch":current.action_epoch,"last_progress_scope":scope,
            "height_progress_pause_reason":None if scope == "grasp_pose" else "active_"+scope+"_obligation"}
        if scope == "grasp_pose":
            correction=grasp_pose_correction_v1(snapshot,snapshot.skill.subject_ref)
            metric=(None if not record.geometry_comparable or correction is None
                or hypot(*correction[:2]) > self.xy_tolerance else abs(correction[2]))
            best,count=self._updated_progress(record.best_observed_height_error_m,
                record.actions_without_new_best,metric)
            changes.update(best_observed_height_error_m=best,actions_without_new_best=count)
        elif scope in ("observation","unknown"):
            complete=record.geometry_comparable and grasp_geometry_observed_v1(snapshot)
            metric=(0. if complete else current.span_deficit_m if record.geometry_comparable else None)
            best,count=self._updated_progress(record.best_observation_deficit_m,
                record.observation_nonprogress_actions,metric,complete=complete)
            changes.update(best_observation_deficit_m=best,observation_nonprogress_actions=count)
        else:
            target=selected_grasp_target_v1(snapshot)
            tcp=snapshot.robot.tcp_position_robot_base_m
            metric=(None if target is None or not grasp_preparation_ready_v1(snapshot)
                else hypot(target[0]-tcp[0],target[1]-tcp[1]))
            best,count=self._updated_progress(record.best_preparation_error_m,
                record.preparation_nonprogress_actions,metric,complete=grasp_height_handoff_ready_v1(snapshot))
            changes.update(best_preparation_error_m=best,preparation_nonprogress_actions=count)
        updated=replace(record,**changes)
        limits=(("grasp_pose",updated.actions_without_new_best),
            ("observation",updated.observation_nonprogress_actions),
            ("preparation",updated.preparation_nonprogress_actions))
        exhausted=next((name for name,count in limits if count >= self.config.maximum_actions_without_new_best),None)
        return replace(updated,blocked_reason=record.blocked_reason or (
            None if exhausted is None else "grasp "+exhausted+" obligation produced no new best confirmed progress"))

    def _updated_progress(self, best, count, metric, *, complete=False):
        if metric is not None and (complete or (best is not None and best-metric >= self.config.minimum_progress_m)):
            return metric if best is None else min(best,metric),0
        # 首次有数值仅建立比较基线，不将证据出现或新选择本身认作运动进展。
        return (metric if best is None else best),count+1

    def select(self, candidate, snapshot):
        if not candidate.choice_id.startswith("prepare_grasp_view_"):
            return snapshot
        r=current_grasp_approach_v1(snapshot)
        if r is not None and r.preparation is not None and r.preparation.choice_id == candidate.choice_id:
            return snapshot
        if r is None or r.stage != "prepare" or not grasp_preparation_ready_v1(snapshot):
            raise ValueError("grasp preparation requires a current recovered observation")
        option=next((x for x in grasp_preparation_options_v1(snapshot,self.xy_tolerance,self.preparation_tolerance)
            if x[0] == candidate.choice_id),None)
        if option is None or candidate.target_robot_base_m != option[1]:
            raise ValueError("grasp preparation choice is not bound to current geometry")
        p=GraspPreparationV1(option[0],snapshot.state_id,snapshot.action_epoch,snapshot.visual_scene.observation_id,
            option[2],r.current,option[3])
        target,tcp=option[1],snapshot.robot.tcp_position_robot_base_m
        self._record=replace(r,preparation=p,
            best_preparation_error_m=hypot(target[0]-tcp[0],target[1]-tcp[1]))
        return replace(snapshot,grasp_approach=self._record)
