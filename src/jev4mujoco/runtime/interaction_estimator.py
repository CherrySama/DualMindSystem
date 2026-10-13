"""从当前 RGB-D 几何和实测运动估计交互，不读取仿真接触信息。"""
from dataclasses import dataclass, replace
from math import sqrt, isfinite

from jev4mujoco.contracts.interaction_evidence import InteractionEvidenceV1
from jev4mujoco.contracts.visual_scene import Visibility, VisualPredicate, RegionType
from jev4mujoco.perception.surface_geometry import polygon_contains_polygon_v1
from jev4mujoco.runtime.tool_geometry import tool_subject_convex_gap_v1


def _norm(vector):
    return sqrt(sum(v * v for v in vector))


def _delta(current, previous):
    return tuple(a - b for a, b in zip(current, previous))


@dataclass(frozen=True, slots=True)
class InteractionEstimatorConfigV1:
    # 估计参数，不能解释为力阈值或任务成功容差。
    geometry_gap_tolerance_m: float = 0.002
    minimum_motion_m: float = 0.0005
    maximum_coupled_delta_error_m: float = 0.006
    maximum_visual_age_s: float = 0.5

    def __post_init__(self):
        if any(not isfinite(value) or value <= 0 for value in (
            self.geometry_gap_tolerance_m, self.minimum_motion_m,
            self.maximum_coupled_delta_error_m, self.maximum_visual_age_s)):
            raise ValueError("interaction estimate parameters must be finite and positive")


class VisualInteractionEstimatorV1:
    def __init__(self, config):
        self.config = config
        self._previous = None

    def reset(self):
        self._previous = None

    def observe(self, robot, scene, subject, tool, recent_action=None):
        base = dict(subject_ref=subject.track_id, state_id=robot.state_id,
            action_epoch=robot.action_epoch, observation_id=scene.observation_id,
            capture_timestamp_s=scene.capture_timestamp_s)
        centroid = subject.geometry.centroid_robot_base_m.value
        extent = subject.geometry.extent_m.value
        footprint = subject.geometry.footprint_xy_robot_base_m.value
        action_end = 0.0 if recent_action is None else recent_action.end_simulation_time_s
        previous = self._previous
        same_scope = previous is not None and previous[0] == (scene.scene_id, subject.track_id)
        if same_scope and (scene.observation_id < previous[1].observation_id
            or scene.capture_timestamp_s < previous[1].capture_timestamp_s
            or (scene.observation_id == previous[1].observation_id
                and (robot.action_epoch != previous[1].action_epoch
                    or robot.tcp_position_robot_base_m != previous[3]))):
            return InteractionEvidenceV1(**base, status="unknown",
                reason="out-of-order image or robot motion without a new synchronized image")
        current = (0 <= robot.timestamp_s - scene.capture_timestamp_s <= self.config.maximum_visual_age_s
            and scene.capture_timestamp_s >= action_end)
        if (not current or subject.visibility is not Visibility.CLEAR or centroid is None
            or extent is None or footprint is None or tool is None):
            return InteractionEvidenceV1(**base, status="unknown",
                reason="current complete RGB-D/tool evidence unavailable")

        # 使用真实标定凸包，避免倾斜夹指的轴对齐盒造成大范围假邻近。
        gap = tool_subject_convex_gap_v1(tool, footprint, centroid, extent)
        if gap is None:
            return InteractionEvidenceV1(**base, status="unknown", reason="convex geometry distance unavailable")
        candidate = gap <= self.config.geometry_gap_tolerance_m
        confidence = min(subject.geometry.centroid_robot_base_m.confidence,
            subject.geometry.extent_m.confidence, subject.geometry.footprint_xy_robot_base_m.confidence)
        result = InteractionEvidenceV1(**base, status="candidate" if candidate else "clear",
            reason="geometry proximity is a contact candidate" if candidate else "current observed volumes are separated",
            tool_subject_gap_m=gap, contact_candidate=candidate, geometry_confidence=confidence)
        if same_scope and scene.observation_id > previous[1].observation_id:
            prior, prior_centroid, prior_tcp = previous[1:]
            tcp_delta = _delta(robot.tcp_position_robot_base_m, prior_tcp)
            subject_delta = _delta(centroid, prior_centroid)
            coupled = None
            if (recent_action is not None and recent_action.executed
                and prior.action_epoch == recent_action.start_action_epoch
                and robot.action_epoch == recent_action.end_action_epoch
                and prior.capture_timestamp_s <= recent_action.start_simulation_time_s
                and _norm(tcp_delta) >= self.config.minimum_motion_m):
                coupled = (_norm(subject_delta) >= self.config.minimum_motion_m
                    and sum(a * b for a, b in zip(tcp_delta, subject_delta)) > 0
                    and _norm(_delta(subject_delta, tcp_delta)) <= self.config.maximum_coupled_delta_error_m)
            continued = (prior.status == "motion_supported" and coupled is None
                and _norm(tcp_delta) < self.config.minimum_motion_m
                and _norm(subject_delta) < self.config.minimum_motion_m)
            result = replace(result, coupled_motion=coupled,
                previous_observation_id=prior.observation_id,
                measured_tcp_delta_m=tcp_delta, observed_subject_delta_m=subject_delta)
            if candidate and (coupled is True or continued):
                result = replace(result, status="motion_supported",
                    reason="observed coupled tool/object motion supports contact estimate" if coupled is True
                    else "estimated continuity from prior coupled motion and current stationary nearby geometry")
        if not same_scope or scene.observation_id > previous[1].observation_id:
            self._previous = ((scene.scene_id, subject.track_id), result, centroid, robot.tcp_position_robot_base_m)
        elif scene.observation_id == previous[1].observation_id:
            # 同帧轮询不能累计新证据，也不能丢失同帧已计算的运动反馈。
            result = replace(previous[1], state_id=robot.state_id, action_epoch=robot.action_epoch)
        return result


def visual_support_estimate_v1(scene, subject, *, current, height_tolerance_m):
    """沿用公开支撑几何容差；区域内关系单独不能证明底面支撑。"""
    centroid, extent = subject.geometry.centroid_robot_base_m.value, subject.geometry.extent_m.value
    footprint = subject.geometry.footprint_xy_robot_base_m.value
    if not current or subject.visibility is not Visibility.CLEAR or centroid is None or extent is None or footprint is None:
        return None, None
    for relation in scene.relations:
        if (relation.subject_ref == subject.track_id and relation.predicate is VisualPredicate.ON_SURFACE
            and relation.value and relation.confidence > 0):
            return True, relation.reference_ref
    bottom = centroid[2] - extent[2] / 2
    observed_surfaces = 0
    for region in scene.regions:
        if region.region_type is RegionType.POINT:
            continue
        boundary = region.geometry.boundary_xy_robot_base_m
        height = (region.geometry.minimum_z_robot_base_m if region.region_type is RegionType.VOLUME_3D
            else region.geometry.surface_height_robot_base_m)
        if boundary is None or boundary.value is None or height is None or height.value is None:
            continue
        observed_surfaces += 1
        if abs(bottom - height.value) <= height_tolerance_m and polygon_contains_polygon_v1(boundary.value, footprint):
            return True, region.region_id
    return (False if observed_surfaces else None), None
