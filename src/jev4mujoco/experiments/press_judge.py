"""隔离的仿真按压事件裁判；只收集真值轨迹，不能决定在线执行。"""
from dataclasses import dataclass
from math import sqrt

from jev4mujoco.contracts.actions import AtomicAction


@dataclass(frozen=True, slots=True)
class ButtonTruthSampleV1:
    timestamp_s: float
    center_robot_base_m: tuple[float, float, float]
    tcp_robot_base_m: tuple[float, float, float]
    effector_contact: bool


class ButtonPressTrajectoryJudgeV1:
    def __init__(self, normal_robot_base, visual_resolution_m):
        length = sqrt(sum(v*v for v in normal_robot_base))
        if length == 0 or visual_resolution_m <= 0:
            raise ValueError("judge requires a normal and positive effect resolution")
        self.normal = tuple(v/length for v in normal_robot_base)
        self.resolution = visual_resolution_m
        self.samples: list[ButtonTruthSampleV1] = []

    def capture(self, raw):
        self.samples.append(ButtonTruthSampleV1(raw.simulation_time_s,
            raw.bowl_position_robot_base_m, raw.tcp_position_robot_base_m,
            raw.contacts.left_finger_bowl_contact or raw.contacts.right_finger_bowl_contact))

    def evaluate(self, action_results):
        """独立核对执行窗口内的物理下压，不读取公开按压事件或Runtime结论。"""
        peak = 0.0
        occurred_at = None
        for action in action_results:
            if not action.executed or action.action is not AtomicAction.DOWN:
                continue
            prior = [s for s in self.samples if s.timestamp_s <= action.start_simulation_time_s + 1e-10]
            if not prior:
                continue
            before = prior[-1]
            for sample in self.samples:
                if not action.start_simulation_time_s < sample.timestamp_s <= action.end_simulation_time_s + 1e-10:
                    continue
                travel = sum((a-b)*n for a,b,n in zip(before.center_robot_base_m, sample.center_robot_base_m, self.normal))
                tcp_travel = sum((a-b)*n for a,b,n in zip(before.tcp_robot_base_m, sample.tcp_robot_base_m, self.normal))
                if sample.effector_contact and tcp_travel > self.resolution:
                    peak = max(peak, travel)
                    if travel > self.resolution and occurred_at is None:
                        occurred_at = sample.timestamp_s
        return occurred_at is not None, peak, occurred_at
