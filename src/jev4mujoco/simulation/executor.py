from __future__ import annotations

from dataclasses import dataclass
from math import acos, ceil, cos, sin
from threading import Lock

import numpy as np
from numpy.typing import NDArray

from jev4mujoco.simulation.backend import MujocoBackend
from jev4mujoco.simulation.config import StageConfig
from jev4mujoco.contracts.actions import (
    ActionRequest,
    ActionResult,
    ActionStatus,
    AtomicAction,
    RawStateSnapshot,
    TRANSLATION_DIRECTIONS_ROBOT_BASE,
)


FloatArray = NDArray[np.float64]


_TRANSLATION_DELTAS: dict[AtomicAction, FloatArray] = {
    action: np.asarray(direction)
    for action, direction in TRANSLATION_DIRECTIONS_ROBOT_BASE.items()
}


def _rotation_vector(rotation: FloatArray) -> FloatArray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = acos(cosine)
    if angle < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    vector = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-10:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64) * angle
    return vector / norm * angle


def _rotation_about_x(angle_rad: float) -> FloatArray:
    cosine = cos(angle_rad)
    sine = sin(angle_rad)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class _ExecutionMetrics:
    end: RawStateSnapshot
    physics_steps: int
    commanded_delta_base: FloatArray
    measured_delta_base: FloatArray
    commanded_joint6_delta: float
    measured_joint6_delta: float
    commanded_gripper: float
    position_error: float
    rotation_error: float
    tcp_drift: float
    gripper_error: float
    success: bool


class AtomicActionExecutor:
    """Executes one discrete JEV action at a time through the MuJoCo backend."""

    def __init__(self, backend: MujocoBackend, config: StageConfig):
        self._backend = backend
        self._config = config
        self._execution_lock = Lock()
        self._last_snapshot = backend.snapshot()

    def execute(self, request: ActionRequest) -> ActionResult:
        if not self._execution_lock.acquire(blocking=False):
            start = self._last_snapshot
            commanded_delta, commanded_joint6_delta, commanded_gripper = self._command(
                request.action, start, request.translation_step_m
            )
            return self._rejected(
                request,
                start,
                commanded_delta,
                commanded_joint6_delta,
                commanded_gripper,
                "another_action_is_in_flight",
            )
        try:
            return self._execute_locked(request)
        finally:
            self._execution_lock.release()

    def _execute_locked(self, request: ActionRequest) -> ActionResult:
        start = self._backend.snapshot()
        self._last_snapshot = start
        commanded_delta, commanded_joint6_delta, commanded_gripper = self._command(
            request.action, start, request.translation_step_m
        )
        if request.based_on_state_id != start.state_id:
            return self._rejected(
                request,
                start,
                commanded_delta,
                commanded_joint6_delta,
                commanded_gripper,
                "stale_state_id",
            )
        if request.based_on_action_epoch != start.action_epoch:
            return self._rejected(
                request,
                start,
                commanded_delta,
                commanded_joint6_delta,
                commanded_gripper,
                "stale_action_epoch",
            )

        try:
            if request.action in _TRANSLATION_DELTAS:
                metrics = self._execute_translation(start, commanded_delta)
            elif request.action in (AtomicAction.ROTATE_CW, AtomicAction.ROTATE_CCW):
                metrics = self._execute_rotation(start, commanded_joint6_delta)
            else:
                metrics = self._execute_gripper_or_hold(start, request.action, commanded_gripper)
        except (ValueError, np.linalg.LinAlgError) as error:
            end = self._backend.snapshot()
            physics_steps = int(
                round(
                    (end.simulation_time_s - start.simulation_time_s)
                    / self._backend.physics_dt_s
                )
            )
            executed = physics_steps > 0
            if executed:
                end = self._backend.commit_action()
            self._last_snapshot = end
            return ActionResult(
                event_type="action_result",
                decision_id=request.decision_id,
                action=request.action,
                based_on_state_id=request.based_on_state_id,
                start_action_epoch=start.action_epoch,
                end_action_epoch=end.action_epoch,
                executed=executed,
                commanded_delta_robot_base_m=tuple(float(v) for v in commanded_delta),  # type: ignore[arg-type]
                measured_delta_robot_base_m=self._measured_delta(start, end),
                measured_tcp_position_robot_base_m=end.tcp_position_robot_base_m,
                measured_tcp_orientation_robot_base_wxyz=end.tcp_orientation_robot_base_wxyz,
                commanded_joint6_delta_rad=commanded_joint6_delta,
                measured_joint6_delta_rad=end.joint_positions_rad[5] - start.joint_positions_rad[5],
                commanded_gripper_joint_m=commanded_gripper,
                measured_gripper_joint_m=end.gripper_joint_m,
                gripper_error_m=abs(end.gripper_joint_m - commanded_gripper),
                position_error_m=float(
                    np.linalg.norm(np.asarray(self._measured_delta(start, end)) - commanded_delta)
                ),
                rotation_error_rad=0.0,
                tcp_drift_m=float(np.linalg.norm(np.asarray(self._measured_delta(start, end)))),
                status=ActionStatus.FAILED,
                reason=str(error),
                physics_steps=physics_steps,
                start_simulation_time_s=start.simulation_time_s,
                end_simulation_time_s=end.simulation_time_s,
            )
        end = self._backend.commit_action()
        self._last_snapshot = end
        status = ActionStatus.COMPLETED if metrics.success else ActionStatus.TIMED_OUT
        reason = None if metrics.success else "acceptance_threshold_not_met_before_timeout"
        return ActionResult(
            event_type="action_result",
            decision_id=request.decision_id,
            action=request.action,
            based_on_state_id=request.based_on_state_id,
            start_action_epoch=start.action_epoch,
            end_action_epoch=end.action_epoch,
            executed=metrics.physics_steps > 0,
            commanded_delta_robot_base_m=tuple(
                float(value) for value in metrics.commanded_delta_base
            ),  # type: ignore[arg-type]
            measured_delta_robot_base_m=tuple(
                float(value) for value in metrics.measured_delta_base
            ),  # type: ignore[arg-type]
            measured_tcp_position_robot_base_m=end.tcp_position_robot_base_m,
            measured_tcp_orientation_robot_base_wxyz=end.tcp_orientation_robot_base_wxyz,
            commanded_joint6_delta_rad=metrics.commanded_joint6_delta,
            measured_joint6_delta_rad=metrics.measured_joint6_delta,
            commanded_gripper_joint_m=metrics.commanded_gripper,
            measured_gripper_joint_m=end.gripper_joint_m,
            gripper_error_m=metrics.gripper_error,
            position_error_m=metrics.position_error,
            rotation_error_rad=metrics.rotation_error,
            tcp_drift_m=metrics.tcp_drift,
            status=status,
            reason=reason,
            physics_steps=metrics.physics_steps,
            start_simulation_time_s=start.simulation_time_s,
            end_simulation_time_s=end.simulation_time_s,
        )

    def _command(
        self, action: AtomicAction, start: RawStateSnapshot,
        translation_step_m: float | None = None,
    ) -> tuple[FloatArray, float, float]:
        delta = np.zeros(3, dtype=np.float64)
        joint6_delta = 0.0
        gripper = start.gripper_target_m
        if action in _TRANSLATION_DELTAS:
            step = self._config.cartesian_step_m if translation_step_m is None else translation_step_m
            if not np.isfinite(step) or not 0.0 < step <= self._config.cartesian_step_m:
                raise ValueError("translation step must be positive and within the configured limit")
            delta = _TRANSLATION_DELTAS[action] * step
        elif action is AtomicAction.ROTATE_CW:
            joint6_delta = self._config.rotation_step_rad
        elif action is AtomicAction.ROTATE_CCW:
            joint6_delta = -self._config.rotation_step_rad
        elif action is AtomicAction.GRIPPER_OPEN:
            gripper = self._config.gripper_open_joint_m
        elif action is AtomicAction.GRIPPER_CLOSE:
            gripper = self._config.gripper_closed_joint_m
        elif action is not AtomicAction.HOLD:
            raise ValueError(f"Unsupported action: {action}")
        return delta, joint6_delta, gripper

    def _dls_update(
        self,
        position_error_world: FloatArray,
        rotation_error_world: FloatArray,
        joint_count: int = 6,
    ) -> FloatArray:
        jacobian_position, jacobian_rotation = self._backend.arm_jacobian_world()
        weight = self._config.orientation_weight_m_per_rad
        jacobian = np.vstack(
            (jacobian_position[:, :joint_count], weight * jacobian_rotation[:, :joint_count])
        )
        error = np.concatenate((position_error_world, weight * rotation_error_world))
        damping = self._config.dls_damping
        update = np.einsum(
            "ji,j->i",
            jacobian,
            np.linalg.solve(
                np.einsum("ik,jk->ij", jacobian, jacobian)
                + damping * damping * np.eye(6),
                error,
            ),
        )
        maximum = float(np.max(np.abs(update)))
        if maximum > self._config.maximum_joint_update_rad:
            update *= self._config.maximum_joint_update_rad / maximum
        return update

    def _execute_translation(
        self, start: RawStateSnapshot, commanded_delta_base: FloatArray
    ) -> _ExecutionMetrics:
        start_position_world = np.asarray(start.tcp_position_world_m)
        target_position_world = (
            start_position_world
            + np.einsum(
                "ij,j->i", self._backend.base_rotation_world(), commanded_delta_base
            )
        )
        target_rotation_world = self._backend.tcp_rotation_world()
        lower, upper = self._backend.arm_joint_limits
        holding_object = (
            np.isclose(start.gripper_target_m, self._config.gripper_closed_joint_m)
            and start.contacts.left_finger_bowl_contact
            and start.contacts.right_finger_bowl_contact
        )
        convergence_fraction = (
            self._config.held_object_convergence_fraction
            if holding_object
            else self._config.free_space_convergence_fraction
        )
        position_convergence_m = (
            self._config.translation_error_m * convergence_fraction
        )
        maximum_steps = ceil(self._config.action_timeout_s / self._backend.physics_dt_s)
        steps = 0
        for _ in range(maximum_steps):
            current = self._backend.snapshot()
            measured_delta = np.asarray(current.tcp_position_robot_base_m) - np.asarray(
                start.tcp_position_robot_base_m
            )
            rotation_error = _rotation_vector(
                np.einsum(
                    "ij,kj->ik",
                    target_rotation_world,
                    self._backend.tcp_rotation_world(),
                )
            )
            if (
                np.linalg.norm(measured_delta - commanded_delta_base)
                <= position_convergence_m
                and np.linalg.norm(rotation_error) <= self._config.rotation_error_rad * 0.25
            ):
                break
            position_error = target_position_world - np.asarray(current.tcp_position_world_m)
            update = self._dls_update(position_error, rotation_error)
            nominal_target = np.clip(
                np.asarray(current.joint_positions_rad) + update, lower, upper
            )
            self._backend.set_arm_compensated_position_targets(nominal_target)
            self._backend.set_gripper_position_target(start.gripper_target_m)
            self._backend.step()
            steps += 1

        end = self._backend.snapshot()
        measured_delta = np.asarray(end.tcp_position_robot_base_m) - np.asarray(
            start.tcp_position_robot_base_m
        )
        position_error = float(np.linalg.norm(measured_delta - commanded_delta_base))
        rotation_error = float(
            np.linalg.norm(
                _rotation_vector(
                    np.einsum(
                        "ij,kj->ik",
                        target_rotation_world,
                        self._backend.tcp_rotation_world(),
                    )
                )
            )
        )
        return _ExecutionMetrics(
            end=end,
            physics_steps=steps,
            commanded_delta_base=commanded_delta_base,
            measured_delta_base=measured_delta,
            commanded_joint6_delta=0.0,
            measured_joint6_delta=end.joint_positions_rad[5] - start.joint_positions_rad[5],
            commanded_gripper=start.gripper_target_m,
            position_error=position_error,
            rotation_error=rotation_error,
            tcp_drift=position_error,
            gripper_error=abs(end.gripper_joint_m - start.gripper_target_m),
            success=(
                position_error <= self._config.translation_error_m
                and rotation_error <= self._config.rotation_error_rad
            ),
        )

    def _execute_rotation(
        self, start: RawStateSnapshot, commanded_joint6_delta: float
    ) -> _ExecutionMetrics:
        start_position_world = np.asarray(start.tcp_position_world_m)
        start_rotation_world = self._backend.tcp_rotation_world()
        target_rotation_world = np.einsum(
            "ij,jk->ik",
            start_rotation_world,
            _rotation_about_x(commanded_joint6_delta),
        )
        lower, upper = self._backend.arm_joint_limits
        joint6_target = start.joint_positions_rad[5] + commanded_joint6_delta
        if joint6_target < lower[5] or joint6_target > upper[5]:
            raise ValueError("joint6 rotation target exceeds joint limit")

        maximum_steps = ceil(self._config.action_timeout_s / self._backend.physics_dt_s)
        steps = 0
        for _ in range(maximum_steps):
            current = self._backend.snapshot()
            position_error = start_position_world - np.asarray(current.tcp_position_world_m)
            rotation_error = _rotation_vector(
                np.einsum(
                    "ij,kj->ik",
                    target_rotation_world,
                    self._backend.tcp_rotation_world(),
                )
            )
            target_axis = target_rotation_world[:, 0]
            non_roll_error = rotation_error - target_axis * float(
                np.dot(rotation_error, target_axis)
            )
            measured_joint6_delta = current.joint_positions_rad[5] - start.joint_positions_rad[5]
            if (
                abs(measured_joint6_delta - commanded_joint6_delta)
                <= self._config.rotation_error_rad * 0.25
                and np.linalg.norm(position_error)
                <= self._config.rotation_tcp_drift_m * 0.25
                and np.linalg.norm(non_roll_error) <= self._config.rotation_error_rad * 0.25
            ):
                break
            update = self._dls_update(position_error, non_roll_error, joint_count=5)
            nominal_target = np.asarray(current.joint_positions_rad).copy()
            nominal_target[:5] = np.clip(
                nominal_target[:5] + update, lower[:5], upper[:5]
            )
            nominal_target[5] = joint6_target
            self._backend.set_arm_compensated_position_targets(nominal_target)
            self._backend.set_gripper_position_target(start.gripper_target_m)
            self._backend.step()
            steps += 1

        end = self._backend.snapshot()
        measured_delta_base = np.asarray(end.tcp_position_robot_base_m) - np.asarray(
            start.tcp_position_robot_base_m
        )
        measured_joint6_delta = end.joint_positions_rad[5] - start.joint_positions_rad[5]
        joint6_error = abs(measured_joint6_delta - commanded_joint6_delta)
        tcp_drift = float(np.linalg.norm(measured_delta_base))
        return _ExecutionMetrics(
            end=end,
            physics_steps=steps,
            commanded_delta_base=np.zeros(3, dtype=np.float64),
            measured_delta_base=measured_delta_base,
            commanded_joint6_delta=commanded_joint6_delta,
            measured_joint6_delta=measured_joint6_delta,
            commanded_gripper=start.gripper_target_m,
            position_error=tcp_drift,
            rotation_error=joint6_error,
            tcp_drift=tcp_drift,
            gripper_error=abs(end.gripper_joint_m - start.gripper_target_m),
            success=(
                joint6_error <= self._config.rotation_error_rad
                and tcp_drift <= self._config.rotation_tcp_drift_m
            ),
        )

    def _execute_gripper_or_hold(
        self,
        start: RawStateSnapshot,
        action: AtomicAction,
        commanded_gripper: float,
    ) -> _ExecutionMetrics:
        start_position_world = np.asarray(start.tcp_position_world_m)
        target_rotation_world = self._backend.tcp_rotation_world()
        lower, upper = self._backend.arm_joint_limits
        duration = (
            self._config.hold_duration_s
            if action is AtomicAction.HOLD
            else self._config.action_timeout_s
        )
        maximum_steps = ceil(duration / self._backend.physics_dt_s)
        steps = 0
        for _ in range(maximum_steps):
            current = self._backend.snapshot()
            position_error = start_position_world - np.asarray(current.tcp_position_world_m)
            rotation_error = _rotation_vector(
                np.einsum(
                    "ij,kj->ik",
                    target_rotation_world,
                    self._backend.tcp_rotation_world(),
                )
            )
            gripper_error = abs(current.gripper_joint_m - commanded_gripper)
            if (
                action is not AtomicAction.HOLD
                and gripper_error <= self._config.gripper_error_m * 0.5
                and np.linalg.norm(position_error)
                <= self._config.translation_error_m * 0.25
                and np.linalg.norm(rotation_error) <= self._config.rotation_error_rad * 0.25
            ):
                break
            update = self._dls_update(position_error, rotation_error)
            nominal_target = np.clip(
                np.asarray(current.joint_positions_rad) + update, lower, upper
            )
            self._backend.set_arm_compensated_position_targets(nominal_target)
            self._backend.set_gripper_position_target(commanded_gripper)
            self._backend.step()
            steps += 1

        end = self._backend.snapshot()
        measured_delta = np.asarray(end.tcp_position_robot_base_m) - np.asarray(
            start.tcp_position_robot_base_m
        )
        tcp_drift = float(np.linalg.norm(measured_delta))
        rotation_error = float(
            np.linalg.norm(
                _rotation_vector(
                    np.einsum(
                        "ij,kj->ik",
                        target_rotation_world,
                        self._backend.tcp_rotation_world(),
                    )
                )
            )
        )
        gripper_error = abs(end.gripper_joint_m - commanded_gripper)
        gripper_completed = (
            action in (AtomicAction.GRIPPER_CLOSE, AtomicAction.HOLD)
            or gripper_error <= self._config.gripper_error_m
        )
        return _ExecutionMetrics(
            end=end,
            physics_steps=steps,
            commanded_delta_base=np.zeros(3, dtype=np.float64),
            measured_delta_base=measured_delta,
            commanded_joint6_delta=0.0,
            measured_joint6_delta=end.joint_positions_rad[5] - start.joint_positions_rad[5],
            commanded_gripper=commanded_gripper,
            position_error=tcp_drift,
            rotation_error=rotation_error,
            tcp_drift=tcp_drift,
            gripper_error=gripper_error,
            success=(
                tcp_drift <= self._config.translation_error_m
                and rotation_error <= self._config.rotation_error_rad
                and gripper_completed
            ),
        )

    @staticmethod
    def _measured_delta(start: RawStateSnapshot, end: RawStateSnapshot) -> tuple[float, float, float]:
        delta = np.asarray(end.tcp_position_robot_base_m) - np.asarray(
            start.tcp_position_robot_base_m
        )
        return tuple(float(value) for value in delta)  # type: ignore[return-value]

    def _rejected(
        self,
        request: ActionRequest,
        state: RawStateSnapshot,
        commanded_delta: FloatArray,
        commanded_joint6_delta: float,
        commanded_gripper: float,
        reason: str,
    ) -> ActionResult:
        return ActionResult(
            event_type="action_result",
            decision_id=request.decision_id,
            action=request.action,
            based_on_state_id=request.based_on_state_id,
            start_action_epoch=state.action_epoch,
            end_action_epoch=state.action_epoch,
            executed=False,
            commanded_delta_robot_base_m=tuple(
                float(value) for value in commanded_delta
            ),  # type: ignore[arg-type]
            measured_delta_robot_base_m=(0.0, 0.0, 0.0),
            measured_tcp_position_robot_base_m=state.tcp_position_robot_base_m,
            measured_tcp_orientation_robot_base_wxyz=state.tcp_orientation_robot_base_wxyz,
            commanded_joint6_delta_rad=commanded_joint6_delta,
            measured_joint6_delta_rad=0.0,
            commanded_gripper_joint_m=commanded_gripper,
            measured_gripper_joint_m=state.gripper_joint_m,
            gripper_error_m=abs(state.gripper_joint_m - commanded_gripper),
            position_error_m=float(np.linalg.norm(commanded_delta)),
            rotation_error_rad=abs(commanded_joint6_delta),
            tcp_drift_m=0.0,
            status=ActionStatus.REJECTED,
            reason=reason,
            physics_steps=0,
            start_simulation_time_s=state.simulation_time_s,
            end_simulation_time_s=state.simulation_time_s,
        )
