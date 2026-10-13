"""释放后的当前分类与原时间绑定的观测证据；不声明任务成功。"""
from dataclasses import dataclass
from math import isfinite

from jev4mujoco.contracts.visual_scene import Vec3


@dataclass(frozen=True, slots=True)
class ReleasedPlacementSampleV1:
    state_id: int
    action_epoch: int
    observation_id: int
    capture_timestamp_s: float
    centers_robot_base_m: tuple[tuple[str, Vec3], ...]

    def __post_init__(self):
        if (self.state_id <= 0 or self.observation_id <= 0 or self.action_epoch < 0
            or not isfinite(self.capture_timestamp_s) or self.capture_timestamp_s < 0):
            raise ValueError("invalid released-placement observation identity")
        names = [ref for ref, _ in self.centers_robot_base_m]
        if not names or len(set(names)) != len(names):
            raise ValueError("released-placement samples require unique object identities")
        for ref, center in self.centers_robot_base_m:
            if not ref or len(center) != 3 or not all(isfinite(v) for v in center):
                raise ValueError("released-placement sample requires finite bound geometry")


@dataclass(frozen=True, slots=True)
class ReleasedPlacementOutcomeV1:
    task_id: str
    skill_instance_id: str
    subject_ref: str
    target_ref: str
    status: str
    reason: str
    based_on_state_id: int
    based_on_action_epoch: int
    based_on_observation_id: int
    timestamp_s: float
    release_state_id: int
    release_action_epoch: int
    release_observation_id: int
    release_timestamp_s: float
    goal_satisfied: bool | None
    support_ref: str | None
    stable_miss_samples: tuple[ReleasedPlacementSampleV1, ...]
    required_observations: int
    waiting_started_s: float | None
    waiting_remaining_s: float | None

    def __post_init__(self):
        if self.status not in ("released", "unknown", "settling", "goal_candidate",
                              "miss_candidate", "stable_goal_miss", "observation_timeout"):
            raise ValueError("unknown released-placement outcome")
        if any(not value for value in (self.task_id, self.skill_instance_id,
                                      self.subject_ref, self.target_ref, self.reason)):
            raise ValueError("released-placement outcome requires its scope")
        if self.required_observations <= 0:
            raise ValueError("released-placement observation count must be positive")
        if (any(type(v) is not int or v <= 0 for v in (self.based_on_state_id,
            self.based_on_observation_id, self.release_state_id, self.release_observation_id))
            or self.based_on_action_epoch < 0 or self.release_action_epoch < 0
            or self.goal_satisfied is not None and type(self.goal_satisfied) is not bool):
            raise ValueError("invalid released-placement identity or goal evidence")
        if (self.release_state_id > self.based_on_state_id
            or self.release_action_epoch > self.based_on_action_epoch
            or self.release_observation_id > self.based_on_observation_id
            or self.release_timestamp_s > self.timestamp_s):
            raise ValueError("release evidence cannot come from the future")
        if any(not isfinite(v) or v < 0 for v in (self.timestamp_s, self.release_timestamp_s)):
            raise ValueError("released-placement times must be nonnegative")
        if (self.waiting_started_s is None) != (self.waiting_remaining_s is None):
            raise ValueError("waiting time and remainder must appear together")
        if self.waiting_started_s is not None and (
            not isfinite(self.waiting_started_s) or self.waiting_started_s < 0
            or self.waiting_started_s > self.timestamp_s
            or not isfinite(self.waiting_remaining_s) or self.waiting_remaining_s < 0):
            raise ValueError("invalid released-placement wait interval")
        if self.status == "stable_goal_miss" and (
            self.goal_satisfied is not False or len(self.stable_miss_samples) < self.required_observations):
            raise ValueError("stable goal miss requires explicit consecutive negative evidence")
        for index, sample in enumerate(self.stable_miss_samples):
            if (self.subject_ref not in dict(sample.centers_robot_base_m)
                or sample.state_id > self.based_on_state_id
                or sample.action_epoch > self.based_on_action_epoch
                or sample.observation_id > self.based_on_observation_id
                or sample.capture_timestamp_s > self.timestamp_s):
                raise ValueError("stable miss sample must retain its bound original evidence")
            if index:
                previous = self.stable_miss_samples[index-1]
                if (sample.state_id <= previous.state_id or sample.observation_id <= previous.observation_id
                    or sample.action_epoch < previous.action_epoch
                    or sample.capture_timestamp_s <= previous.capture_timestamp_s):
                    raise ValueError("stable miss samples must be distinct ordered observations")
