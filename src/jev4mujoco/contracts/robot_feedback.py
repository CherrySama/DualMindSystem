"""在线机器人反馈白名单；没有物体位姿、碰撞对或接触力字段。"""
from dataclasses import dataclass
from math import isfinite

from jev4mujoco.contracts.actions import Joint6, QuatWxyz, Vec3


@dataclass(frozen=True, slots=True)
class RobotFeedbackV1:
    state_id: int
    action_epoch: int
    timestamp_s: float
    joint_positions_rad: Joint6
    joint_velocities_rad_s: Joint6
    gripper_joint_m: float
    gripper_target_m: float
    tcp_position_world_m: Vec3
    tcp_position_robot_base_m: Vec3
    tcp_orientation_robot_base_wxyz: QuatWxyz
    right_gripper_joint_m: float | None = None
    estimated_joint_torques_nm: Joint6 | None = None

    def __post_init__(self):
        if type(self.state_id) is not int or type(self.action_epoch) is not int or self.state_id <= 0 or self.action_epoch < 0:
            raise ValueError("robot feedback requires a current sample identity")
        for values, length in ((self.joint_positions_rad, 6), (self.joint_velocities_rad_s, 6),
            (self.tcp_position_world_m, 3), (self.tcp_position_robot_base_m, 3),
            (self.tcp_orientation_robot_base_wxyz, 4)):
            if len(values) != length or not all(isfinite(v) for v in values):
                raise ValueError("robot feedback vectors must have their declared length and finite values")
        if (not isfinite(self.timestamp_s) or self.timestamp_s < 0
            or not all(isfinite(v) for v in (self.gripper_joint_m, self.gripper_target_m))):
            raise ValueError("invalid robot measurement timestamp or gripper feedback")
        if self.right_gripper_joint_m is not None and not isfinite(self.right_gripper_joint_m):
            raise ValueError("independent right finger feedback must be finite or unavailable")
        if self.estimated_joint_torques_nm is not None and (
            len(self.estimated_joint_torques_nm) != 6
            or not all(isfinite(v) for v in self.estimated_joint_torques_nm)):
            raise ValueError("estimated joint torques must contain six finite values")

    @property
    def simulation_time_s(self):
        """兼容现有执行日志命名；反馈时间不依赖仿真物体状态。"""
        return self.timestamp_s


def robot_feedback_from_raw_v1(raw):
    """仿真适配器只复制现实可测机器人信息，不读取 raw.contacts。"""
    return RobotFeedbackV1(raw.state_id, raw.action_epoch, raw.simulation_time_s,
        raw.joint_positions_rad, raw.joint_velocities_rad_s,
        raw.gripper_joint_m, raw.gripper_target_m, raw.tcp_position_world_m,
        raw.tcp_position_robot_base_m, raw.tcp_orientation_robot_base_wxyz,
        raw.right_gripper_joint_m)
