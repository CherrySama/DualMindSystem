"""公开按压事件：历史发生证据不冒充当前接触或当前按钮位置。"""
from dataclasses import dataclass
from math import isfinite

from jev4mujoco.contracts.visual_scene import Vec3


@dataclass(frozen=True, slots=True)
class PressVisualSampleV1:
    state_id: int
    action_epoch: int
    observation_id: int
    capture_timestamp_s: float
    scene_id: str
    sensor_id: str
    calibration_id: str
    top_robot_base_m: Vec3
    centroid_robot_base_m: Vec3
    tcp_robot_base_m: Vec3
    point_robot_base_m: Vec3
    normal_robot_base: Vec3

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.state_id, self.observation_id)):
            raise ValueError("press samples require positive state/observation identities")
        if type(self.action_epoch) is not int or self.action_epoch < 0:
            raise ValueError("press sample epoch must be nonnegative")
        if not isfinite(self.capture_timestamp_s) or self.capture_timestamp_s < 0:
            raise ValueError("press sample time must be finite and nonnegative")
        if any(not isinstance(v, str) or not v for v in (self.scene_id, self.sensor_id, self.calibration_id)):
            raise ValueError("press sample requires scene, sensor and calibration identities")
        for vector in (self.top_robot_base_m, self.centroid_robot_base_m,
                       self.tcp_robot_base_m, self.point_robot_base_m, self.normal_robot_base):
            if len(vector) != 3 or not all(isfinite(v) for v in vector):
                raise ValueError("press sample geometry must be finite three-vectors")
        if abs(sum(v*v for v in self.normal_robot_base) - 1.0) > 1e-6:
            raise ValueError("press normal must be normalized")


@dataclass(frozen=True, slots=True)
class PressOccurrenceV1:
    before: PressVisualSampleV1
    after: PressVisualSampleV1
    decision_id: int
    action_status: str
    button_displacement_m: float
    centroid_displacement_m: float
    tcp_displacement_m: float
    visual_resolution_m: float

    def __post_init__(self):
        a, b = self.before, self.after
        if (b.state_id <= a.state_id or b.observation_id <= a.observation_id
            or b.action_epoch != a.action_epoch + 1
            or b.capture_timestamp_s <= a.capture_timestamp_s
            or (a.scene_id, a.sensor_id, a.calibration_id, a.point_robot_base_m, a.normal_robot_base)
            != (b.scene_id, b.sensor_id, b.calibration_id, b.point_robot_base_m, b.normal_robot_base)):
            raise ValueError("press occurrence requires comparable ordered observations")
        if type(self.decision_id) is not int or self.decision_id <= 0:
            raise ValueError("press occurrence requires its executed decision")
        if self.action_status not in ("completed", "timed_out"):
            raise ValueError("press occurrence must retain the actual action status")
        if not isfinite(self.visual_resolution_m) or self.visual_resolution_m <= 0:
            raise ValueError("press visual resolution must be positive")
        if any(not isfinite(v) or v <= self.visual_resolution_m for v in (
            self.button_displacement_m, self.centroid_displacement_m, self.tcp_displacement_m)):
            raise ValueError("press occurrence must exceed visual resolution")
        for first, second, declared in (
            (a.top_robot_base_m, b.top_robot_base_m, self.button_displacement_m),
            (a.centroid_robot_base_m, b.centroid_robot_base_m, self.centroid_displacement_m),
            (a.tcp_robot_base_m, b.tcp_robot_base_m, self.tcp_displacement_m)):
            actual = sum((x-y)*n for x,y,n in zip(first, second, b.normal_robot_base))
            if abs(actual-declared) > 1e-9:
                raise ValueError("press displacement must match its recorded observations")


@dataclass(frozen=True, slots=True)
class PressEventEvidenceV1:
    task_id: str
    skill_instance_id: str
    subject_ref: str
    reference_ref: str
    status: str
    reason: str
    visual_resolution_m: float
    current_sample: PressVisualSampleV1 | None = None
    previous_sample: PressVisualSampleV1 | None = None
    current_button_displacement_m: float | None = None
    occurrence: PressOccurrenceV1 | None = None
    source: str = "rgbd_robot_action_response"

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (
            self.task_id, self.skill_instance_id, self.subject_ref, self.reference_ref, self.reason)):
            raise ValueError("press evidence requires bound task/skill/object identities")
        if self.status not in ("unknown", "not_observed", "observed"):
            raise ValueError("unknown press evidence status")
        if (self.status == "observed") != (self.occurrence is not None):
            raise ValueError("observed press evidence requires its historical occurrence")
        if not isfinite(self.visual_resolution_m) or self.visual_resolution_m <= 0:
            raise ValueError("press evidence requires a positive visual resolution")
        if self.current_button_displacement_m is not None and not isfinite(self.current_button_displacement_m):
            raise ValueError("current press displacement must be finite or unknown")
        if self.source != "rgbd_robot_action_response":
            raise ValueError("press evidence source is fixed to public visual action response")
