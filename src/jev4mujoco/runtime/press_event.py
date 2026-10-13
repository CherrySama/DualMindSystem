"""仅使用新鲜RGB-D、机器人运动和已执行动作确认按压事件。"""
from dataclasses import replace
from math import hypot, sqrt, isfinite

from jev4mujoco.contracts.actions import AtomicAction, ActionStatus
from jev4mujoco.contracts.press_event import (
    PressEventEvidenceV1, PressOccurrenceV1, PressVisualSampleV1,
)
from jev4mujoco.contracts.visual_scene import RegionType, Visibility


def press_event_observed_v1(snapshot, subject_ref, reference_ref):
    evidence = snapshot.press_event
    if evidence is None or (evidence.task_id, evidence.skill_instance_id,
        evidence.subject_ref, evidence.reference_ref) != (
        snapshot.task.task_id, snapshot.skill.skill_instance_id, subject_ref, reference_ref):
        return None
    event = evidence.occurrence
    if event is not None:
        if (event.after.state_id > snapshot.state_id or event.after.action_epoch > snapshot.action_epoch
            or event.after.observation_id > snapshot.visual_scene.observation_id
            or event.after.capture_timestamp_s > snapshot.timestamp_s
            or (event.after.scene_id, event.after.sensor_id, event.after.calibration_id) != (
                snapshot.visual_scene.scene_id, snapshot.visual_scene.sensor.sensor_id,
                snapshot.visual_scene.sensor.calibration_id)):
            return None
        return True
    return None if evidence.status == "unknown" else False


def _inward_displacement(before, after, normal):
    return sum((a-b)*n for a, b, n in zip(before, after, normal))


class PressEventRuntimeV1:
    def __init__(self, visual_resolution_m, xy_tolerance_m, maximum_visual_age_s):
        if any(not isfinite(v) or v <= 0 for v in (visual_resolution_m, xy_tolerance_m, maximum_visual_age_s)):
            raise ValueError("press measurement bounds must be finite and positive")
        self.resolution = visual_resolution_m
        self.xy_tolerance = xy_tolerance_m
        self.maximum_age = maximum_visual_age_s
        self.reset()

    def reset(self):
        self._key = None
        self._previous = None
        self._occurrence = None

    def _sample(self, snapshot):
        scene = snapshot.visual_scene
        if (not snapshot.freshness.visual_is_fresh or snapshot.freshness.visual_age_s > self.maximum_age
            or scene.capture_timestamp_s > snapshot.timestamp_s
            or snapshot.sources.robot_measurement_timestamp_s != snapshot.timestamp_s
            or (snapshot.recent_action is not None
                and scene.capture_timestamp_s < snapshot.recent_action.end_timestamp_s)):
            return None
        entity = next((e for e in scene.entities if e.track_id == snapshot.skill.subject_ref), None)
        point = next((r for r in scene.regions if r.region_id == snapshot.skill.target_ref), None)
        if (entity is None or entity.visibility is not Visibility.CLEAR or point is None
            or point.region_type is not RegionType.POINT or point.owner_entity_ref != entity.track_id):
            return None
        top, center = entity.geometry.top_center_robot_base_m.value, entity.geometry.centroid_robot_base_m.value
        location, normal = point.geometry.point_robot_base_m, point.geometry.surface_normal_robot_base
        if top is None or center is None or location is None or location.value is None or normal is None or normal.value is None:
            return None
        length = sqrt(sum(v*v for v in normal.value))
        if length == 0:
            return None
        return PressVisualSampleV1(snapshot.state_id, snapshot.action_epoch, scene.observation_id,
            scene.capture_timestamp_s, scene.scene_id, scene.sensor.sensor_id, scene.sensor.calibration_id,
            top, center, snapshot.robot.tcp_position_robot_base_m, location.value,
            tuple(v/length for v in normal.value))

    def attach(self, snapshot, action=None):
        skill = snapshot.skill
        if skill.skill_kind != "press" or skill.subject_ref is None or skill.target_ref is None:
            return snapshot
        key = (snapshot.task.task_id, skill.skill_instance_id, skill.subject_ref, skill.target_ref)
        if key != self._key:
            self.reset()
            self._key = key
        current, previous = self._sample(snapshot), self._previous
        displacement = None
        status, reason = "unknown", "current comparable visual geometry is unavailable"
        if current is not None:
            status, reason = "not_observed", "no executed downward response above visual resolution"
        if (current is not None and previous is not None and action is not None
            and skill.phase == "press_button" and action.executed
            and action.action is AtomicAction.DOWN
            and action.status in (ActionStatus.COMPLETED, ActionStatus.TIMED_OUT)
            and action.based_on_state_id == previous.state_id
            and action.start_action_epoch == previous.action_epoch
            and action.end_action_epoch == current.action_epoch == previous.action_epoch + 1
            and current.state_id > previous.state_id and current.observation_id > previous.observation_id
            and previous.capture_timestamp_s <= action.start_simulation_time_s
            and current.capture_timestamp_s >= action.end_simulation_time_s
            and current.capture_timestamp_s > previous.capture_timestamp_s
            and (current.scene_id, current.sensor_id, current.calibration_id,
                current.point_robot_base_m, current.normal_robot_base) == (
                previous.scene_id, previous.sensor_id, previous.calibration_id,
                previous.point_robot_base_m, previous.normal_robot_base)):
            normal = current.normal_robot_base
            displacement = _inward_displacement(previous.top_robot_base_m, current.top_robot_base_m, normal)
            center_delta = _inward_displacement(previous.centroid_robot_base_m, current.centroid_robot_base_m, normal)
            tcp_delta = _inward_displacement(previous.tcp_robot_base_m, current.tcp_robot_base_m, normal)
            delta = tuple(a-b for a,b in zip(previous.centroid_robot_base_m, current.centroid_robot_base_m))
            lateral = sqrt(sum((d-center_delta*n)**2 for d,n in zip(delta, normal)))
            aligned = all(hypot(s.tcp_robot_base_m[0]-s.point_robot_base_m[0],
                s.tcp_robot_base_m[1]-s.point_robot_base_m[1]) <= self.xy_tolerance for s in (previous, current))
            if (aligned and min(displacement, center_delta, tcp_delta) > self.resolution
                and abs(displacement-center_delta) <= self.resolution
                and abs(center_delta-tcp_delta) <= self.resolution and lateral <= self.resolution
                and self._occurrence is None):
                self._occurrence = PressOccurrenceV1(previous, current, action.decision_id,
                    action.status.value, displacement, center_delta, tcp_delta, self.resolution)
        if self._occurrence is not None:
            status, reason = "observed", "historical executed press response confirmed; current contact is independent"
        evidence = PressEventEvidenceV1(*key, status, reason, self.resolution,
            current, previous, displacement, self._occurrence)
        # 不以陈旧/重复图像刷新动作前观测。阶段转换重用同一新鲜观测时只更新机器人身份。
        if current is not None and (previous is None or current.observation_id > previous.observation_id
            or (current.observation_id == previous.observation_id and current.action_epoch == previous.action_epoch)):
            self._previous = current
        return replace(snapshot, press_event=evidence)
