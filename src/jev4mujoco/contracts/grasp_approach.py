"""抓取接近记录；历史观测及机器人回撤点不充当当前物体几何。"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class GraspViewObservationV1:
    state_id: int
    action_epoch: int
    observation_id: int
    capture_timestamp_s: float
    scene_id: str
    sensor_id: str
    calibration_id: str
    tcp_robot_base_m: tuple[float, float, float]
    centroid_robot_base_m: tuple[float, float, float] | None
    extent_m: tuple[float, float, float] | None
    span_deficit_m: float | None
    visibility: str

    def __post_init__(self):
        if (self.state_id <= 0 or self.action_epoch < 0 or self.observation_id <= 0
            or not isfinite(self.capture_timestamp_s) or self.capture_timestamp_s < 0
            or any(not x for x in (self.scene_id, self.sensor_id, self.calibration_id))
            or self.visibility not in ("clear", "partial", "not_detected")):
            raise ValueError("grasp observation requires bound finite identities")
        for vector in (self.tcp_robot_base_m, self.centroid_robot_base_m, self.extent_m):
            if vector is not None and (len(vector) != 3 or not all(isfinite(v) for v in vector)):
                raise ValueError("grasp observation vectors must be finite or unavailable")
        if self.span_deficit_m is not None and (not isfinite(self.span_deficit_m) or self.span_deficit_m < 0):
            raise ValueError("visibility deficit must be nonnegative or unknown")


@dataclass(frozen=True, slots=True)
class GraspPreparationV1:
    choice_id: str
    selected_state_id: int
    selected_action_epoch: int
    selected_observation_id: int
    offset_xy_m: tuple[float, float]
    origin: GraspViewObservationV1
    tolerance_m: float

    def __post_init__(self):
        if (not self.choice_id.startswith("prepare_grasp_view_")
            or (self.selected_state_id, self.selected_action_epoch, self.selected_observation_id)
            != (self.origin.state_id, self.origin.action_epoch, self.origin.observation_id)
            or self.origin.visibility != "clear" or self.origin.centroid_robot_base_m is None
            or len(self.offset_xy_m) != 2 or not all(isfinite(v) for v in self.offset_xy_m)
            or not isfinite(self.tolerance_m) or self.tolerance_m <= 0):
            raise ValueError("grasp preparation requires an explicit choice and fresh complete origin")


@dataclass(frozen=True, slots=True)
class GraspApproachV1:
    task_id: str
    skill_instance_id: str
    subject_ref: str
    source_ref: str
    current: GraspViewObservationV1
    return_observation: GraspViewObservationV1
    failures: tuple[GraspViewObservationV1, ...]
    failed_preparations: tuple[str, ...]
    stage: str
    reason: str
    xy_tolerance_m: float
    retraction_margin_m: float
    preparation_tolerance_m: float
    preparation: GraspPreparationV1 | None = None
    initial_observation: GraspViewObservationV1 | None = None
    best_observed_height_error_m: float | None = None
    actions_without_new_best: int = 0
    last_counted_action_epoch: int = 0
    geometry_comparable: bool = True
    blocked_reason: str | None = None
    failed_preparation_records: tuple[GraspPreparationV1, ...] = ()
    best_observation_deficit_m: float | None = None
    observation_nonprogress_actions: int = 0
    best_preparation_error_m: float | None = None
    preparation_nonprogress_actions: int = 0
    last_progress_scope: str | None = None
    height_progress_pause_reason: str | None = None

    def __post_init__(self):
        if (any(not x for x in (self.task_id, self.skill_instance_id, self.subject_ref, self.source_ref, self.reason))
            or self.stage not in ("observe", "prepare", "approach", "unknown")
            or not self.failures or self.return_observation.visibility != "clear"
            or any(type(x) is not int or x < 0 for x in (self.actions_without_new_best,
                self.last_counted_action_epoch, self.observation_nonprogress_actions,
                self.preparation_nonprogress_actions))
            or self.last_progress_scope not in (None, "observation", "preparation", "grasp_pose", "unknown")):
            raise ValueError("grasp approach requires bound history and a valid stage")
        if any(not isfinite(v) or v <= 0 for v in (
            self.xy_tolerance_m, self.retraction_margin_m, self.preparation_tolerance_m)):
            raise ValueError("grasp approach requires original geometry and execution tolerances")
        if self.best_observed_height_error_m is not None and (
            not isfinite(self.best_observed_height_error_m) or self.best_observed_height_error_m < 0):
            raise ValueError("observed grasp error must be nonnegative or unknown")
        if any(x.action_epoch > self.current.action_epoch or x.observation_id > self.current.observation_id
            or x.capture_timestamp_s > self.current.capture_timestamp_s for x in self.failures):
            raise ValueError("grasp failure history cannot come from the future")
        if self.initial_observation is not None and (self.initial_observation.visibility != "clear"
            or self.initial_observation.capture_timestamp_s > self.current.capture_timestamp_s):
            raise ValueError("grasp comparison origin must be an earlier complete observation")
        if any(x is not None and (not isfinite(x) or x < 0) for x in (
            self.best_observation_deficit_m, self.best_preparation_error_m)):
            raise ValueError("grasp subgoal progress must be nonnegative or unavailable")
        if any(p.selected_action_epoch > self.current.action_epoch
            or p.origin.capture_timestamp_s > self.current.capture_timestamp_s for p in self.failed_preparation_records):
            raise ValueError("failed preparation selection cannot come from the future")
