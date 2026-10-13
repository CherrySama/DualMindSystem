"""视觉与机器人运动形成的交互估计；不是触觉或力测量。"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class InteractionEvidenceV1:
    subject_ref: str
    state_id: int
    action_epoch: int
    observation_id: int
    capture_timestamp_s: float
    status: str
    reason: str
    tool_subject_gap_m: float | None = None
    contact_candidate: bool | None = None
    coupled_motion: bool | None = None
    previous_observation_id: int | None = None
    measured_tcp_delta_m: tuple[float, float, float] | None = None
    observed_subject_delta_m: tuple[float, float, float] | None = None
    geometry_confidence: float | None = None
    # 尚无校准过的学习接触概率，也没有可用力估计器。
    contact_probability: float | None = None
    force_estimate_n: tuple[float, float, float] | None = None
    source: str = "rgbd_robot_motion_estimate"

    def __post_init__(self):
        if (not self.subject_ref or self.state_id <= 0 or self.action_epoch < 0
            or self.observation_id <= 0 or self.status not in ("unknown", "clear", "candidate", "motion_supported")
            or not isfinite(self.capture_timestamp_s) or self.capture_timestamp_s < 0):
            raise ValueError("interaction estimate requires current evidence and a known status")
        for value in (self.geometry_confidence, self.contact_probability):
            if value is not None and (not isfinite(value) or not 0 <= value <= 1):
                raise ValueError("evidence confidence must be finite in [0, 1] or uncalibrated")
        if self.tool_subject_gap_m is not None and (
            not isfinite(self.tool_subject_gap_m) or self.tool_subject_gap_m < 0):
            raise ValueError("geometry gap must be finite and nonnegative")
        for vector in (self.measured_tcp_delta_m, self.observed_subject_delta_m, self.force_estimate_n):
            if vector is not None and (len(vector) != 3 or not all(isfinite(v) for v in vector)):
                raise ValueError("interaction vectors must contain three finite values")
        if self.previous_observation_id is not None and self.previous_observation_id >= self.observation_id:
            raise ValueError("motion evidence must use an earlier observation")
        if any(value is not None and type(value) is not bool
            for value in (self.contact_candidate, self.coupled_motion)):
            raise ValueError("interaction observations must be Boolean or unknown")
        if (self.status == "clear" and self.contact_candidate is not False
            or self.status in ("candidate", "motion_supported") and self.contact_candidate is not True
            or self.status == "motion_supported" and self.previous_observation_id is None):
            raise ValueError("interaction status must agree with its evidence")
