"""公开观测的释放结果交接；不选动作、不复位、不放宽成功判定。"""
from math import dist

from jev4mujoco.contracts.released_placement import (
    ReleasedPlacementOutcomeV1, ReleasedPlacementSampleV1,
)
from jev4mujoco.contracts.state_snapshot import GraspState
from jev4mujoco.contracts.visual_scene import Visibility
from jev4mujoco.runtime.observation_geometry import task_observation_geometry_v1
from jev4mujoco.runtime.tool_geometry import tool_transfer_clearance_v1


POST_RELEASE_PHASES = ("retract_after_release", "observe_task_geometry", "verify_placement")


class ReleasedPlacementRuntimeV1:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self._scope = None
        self._release = None
        self._release_valid = False
        self._samples = ()
        self._waiting_started_s = None
        self._last_state_id = 0
        self._last_epoch = 0
        self._last_timestamp_s = 0.0
        self._last_observation_id = 0
        self._last_capture_s = -1.0

    def end_observation_wait(self):
        self._waiting_started_s = None

    def _centers(self, snapshot):
        """完整主体/目标几何，返回所有需核对稳定性的实体中心。"""
        skill, scene = snapshot.skill, snapshot.visual_scene
        entities = {e.track_id: e for e in scene.entities}
        refs = [skill.subject_ref]
        region = next((r for r in scene.regions if r.region_id == skill.target_ref), None)
        if region is None:
            refs.append(skill.target_ref)
        else:
            boundary = region.geometry.boundary_xy_robot_base_m
            height = region.geometry.surface_height_robot_base_m
            if boundary is None or boundary.value is None or height is None or height.value is None:
                return None
            if region.owner_entity_ref is not None:
                refs.append(region.owner_entity_ref)
        centers = []
        for ref in dict.fromkeys(refs):
            entity = entities.get(ref)
            if entity is None or entity.visibility is not Visibility.CLEAR:
                return None
            g = entity.geometry
            if any(value.value is None for value in (
                g.centroid_robot_base_m, g.top_center_robot_base_m, g.extent_m,
                g.footprint_xy_robot_base_m)):
                return None
            centers.append((ref, g.centroid_robot_base_m.value))
        return tuple(centers)

    def observe(self, snapshot, *, visual_current, robot_current):
        skill, scene, physical = snapshot.skill, snapshot.visual_scene, snapshot.physical
        if (skill.skill_kind != "place" or skill.goal is None or skill.subject_ref is None
            or skill.target_ref is None or skill.goal.reference_ref != skill.target_ref):
            self.reset()
            return None
        scope = (snapshot.task.task_id, skill.skill_instance_id, skill.subject_ref,
            skill.source_ref, skill.target_ref, skill.goal, scene.scene_id,
            scene.sensor.sensor_id, scene.sensor.calibration_id)
        if (scope != self._scope or snapshot.state_id < self._last_state_id
            or snapshot.action_epoch < self._last_epoch or snapshot.timestamp_s < self._last_timestamp_s
            or scene.observation_id < self._last_observation_id or scene.capture_timestamp_s < self._last_capture_s):
            self.reset()
            self._scope = scope
        self._last_state_id, self._last_epoch = snapshot.state_id, snapshot.action_epoch
        self._last_timestamp_s = snapshot.timestamp_s
        new_observation = (scene.observation_id > self._last_observation_id
            and scene.capture_timestamp_s > self._last_capture_s)
        self._last_observation_id = max(self._last_observation_id, scene.observation_id)
        self._last_capture_s = max(self._last_capture_s, scene.capture_timestamp_s)
        current = (visual_current and robot_current
            and snapshot.sources.visual_observation_id == scene.observation_id
            and snapshot.sources.visual_scene_id == scene.scene_id
            and snapshot.sources.visual_capture_timestamp_s == scene.capture_timestamp_s)
        open_observed = (robot_current and abs(snapshot.robot.gripper.opening_m
            - self.config.gripper_opening_m) <= self.config.gripper_tolerance_m)
        release_observed = current and open_observed and physical.grasp_state is GraspState.NOT_HELD
        if current and physical.grasp_state in (GraspState.HELD, GraspState.CANDIDATE_HELD):
            # 历史打开事件仍保留原时间；重新持有后不能继承其当前有效性。
            self._release_valid = False
            self._samples = ()
        action = snapshot.recent_action
        if (skill.phase == "lower_to_release" and release_observed and action is not None
            and action.action == "gripper_open" and action.status in ("completed", "timed_out")
            and action.end_action_epoch == snapshot.action_epoch
            and (self._release is None or not self._release_valid)):
            self._release = (snapshot.state_id, snapshot.action_epoch,
                scene.observation_id, snapshot.timestamp_s)
            self._release_valid = True
        released = release_observed and self._release_valid
        if self._release is None or skill.phase not in ("lower_to_release", *POST_RELEASE_PHASES):
            return None
        relation = next((r for r in scene.relations if r.subject_ref == skill.subject_ref
            and r.reference_ref == skill.target_ref and r.predicate is skill.goal.predicate), None)
        centers = self._centers(snapshot) if current else None
        goal = (None if centers is None or relation is None or relation.value is None
            else relation.value is skill.goal.desired_value)
        support = physical.support_state
        supported = (None if not current or support.subject_ref != skill.subject_ref
            else support.supported)
        geometry = tool_transfer_clearance_v1(snapshot) if current else None
        clearance = None if geometry is None else geometry.clearance_m
        at_clearance = clearance is not None and clearance >= self.config.placement_obstacle_clearance_m
        motion_ready = released and supported is True and goal is True and clearance is not None
        if skill.phase == "observe_task_geometry":
            observation = task_observation_geometry_v1(snapshot)
            motion_ready = (motion_ready and at_clearance and observation.geometry_observed is False
                and observation.span_deficit_m is not None and observation.robot_occlusion_hint is True)
        elif skill.phase != "retract_after_release":
            motion_ready = False

        status, reason = "unknown", "current release, support, goal or tool evidence is unavailable"
        if skill.phase == "lower_to_release" and released:
            status, reason = "released", "opening and absence of holding observed; placement remains unverified"
        elif released and supported is False:
            status, reason = "settling", "current support is not established; wait for comparable observations"
        elif released and supported is True and goal is True:
            status, reason = "goal_candidate", "current goal matches; original exit and stable verification remain required"
        elif released and supported is True and goal is False and at_clearance:
            status, reason = "miss_candidate", "current supported placement misses the bound goal; stability not yet confirmed"

        if status != "miss_candidate":
            self._samples = ()
        elif new_observation:
            sample = ReleasedPlacementSampleV1(snapshot.state_id, snapshot.action_epoch,
                scene.observation_id, scene.capture_timestamp_s, centers)
            last = self._samples[-1] if self._samples else None
            if last is None:
                self._samples = (sample,)
            else:
                previous = dict(last.centers_robot_base_m)
                stable = (previous.keys() == dict(centers).keys() and all(
                    dist(center, previous[ref]) <= self.config.placement_maximum_centroid_shift_m
                    for ref, center in centers))
                if not stable:
                    self._samples = (sample,)
                elif sample.capture_timestamp_s - last.capture_timestamp_s + 1e-12 >= self.config.placement_minimum_observation_interval_s:
                    self._samples = (*self._samples, sample)
        if len(self._samples) >= self.config.placement_required_consecutive_observations:
            status, reason = "stable_goal_miss", "fresh stable released placement misses the bound goal"

        if skill.phase == "lower_to_release" or motion_ready:
            self._waiting_started_s = None
        elif self._waiting_started_s is None:
            self._waiting_started_s = snapshot.timestamp_s
        remaining = (None if self._waiting_started_s is None else max(0.0,
            self.config.maximum_pending_verification_s - snapshot.timestamp_s + self._waiting_started_s))
        if status != "stable_goal_miss" and remaining is not None and remaining <= 1e-12:
            status, reason = "observation_timeout", "released placement exhausted observation wait; task effect remains unconfirmed"
        return ReleasedPlacementOutcomeV1(
            snapshot.task.task_id, skill.skill_instance_id, skill.subject_ref, skill.target_ref,
            status, reason, snapshot.state_id, snapshot.action_epoch, scene.observation_id,
            snapshot.timestamp_s, *self._release, goal,
            support.support_region_ref if supported is True else None,
            self._samples, self.config.placement_required_consecutive_observations,
            self._waiting_started_s, remaining)
