from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]
Joint6 = tuple[float, float, float, float, float, float]


class AtomicAction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"
    GRIPPER_OPEN = "gripper_open"
    GRIPPER_CLOSE = "gripper_close"
    HOLD = "hold"


TRANSLATION_DIRECTIONS_ROBOT_BASE: dict[AtomicAction, Vec3] = {
    AtomicAction.FORWARD: (1.0, 0.0, 0.0),
    AtomicAction.BACKWARD: (-1.0, 0.0, 0.0),
    AtomicAction.LEFT: (0.0, 1.0, 0.0),
    AtomicAction.RIGHT: (0.0, -1.0, 0.0),
    AtomicAction.UP: (0.0, 0.0, 1.0),
    AtomicAction.DOWN: (0.0, 0.0, -1.0),
}


class ActionStatus(str, Enum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class PhysicalContactSnapshot:
    left_finger_bowl_contact: bool
    right_finger_bowl_contact: bool
    left_finger_bowl_normal_force_n: float
    right_finger_bowl_normal_force_n: float
    bowl_table_contact: bool
    bowl_table_normal_force_n: float


@dataclass(frozen=True, slots=True)
class ActionRequest:
    decision_id: int
    based_on_state_id: int
    based_on_action_epoch: int
    action: AtomicAction
    translation_step_m: float | None = None


@dataclass(frozen=True, slots=True)
class RawStateSnapshot:
    state_id: int
    action_epoch: int
    simulation_time_s: float
    joint_positions_rad: Joint6
    joint_velocities_rad_s: Joint6
    gripper_joint_m: float
    gripper_target_m: float
    tcp_position_world_m: Vec3
    tcp_position_robot_base_m: Vec3
    tcp_orientation_robot_base_wxyz: QuatWxyz
    bowl_position_world_m: Vec3
    bowl_position_robot_base_m: Vec3
    bowl_orientation_robot_base_wxyz: QuatWxyz
    bowl_linear_velocity_world_mps: Vec3
    bowl_angular_velocity_world_radps: Vec3
    target_position_robot_base_m: Vec3 | None
    contacts: PhysicalContactSnapshot
    right_gripper_joint_m: float | None = None


@dataclass(frozen=True, slots=True)
class ActionResult:
    event_type: str
    decision_id: int
    action: AtomicAction
    based_on_state_id: int
    start_action_epoch: int
    end_action_epoch: int
    executed: bool
    commanded_delta_robot_base_m: Vec3
    measured_delta_robot_base_m: Vec3
    measured_tcp_position_robot_base_m: Vec3
    measured_tcp_orientation_robot_base_wxyz: QuatWxyz
    commanded_joint6_delta_rad: float
    measured_joint6_delta_rad: float
    commanded_gripper_joint_m: float
    measured_gripper_joint_m: float
    gripper_error_m: float
    position_error_m: float
    rotation_error_rad: float
    tcp_drift_m: float
    status: ActionStatus
    reason: str | None
    physics_steps: int
    start_simulation_time_s: float
    end_simulation_time_s: float
