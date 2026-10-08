from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from jev4mujoco import ActionRequest, ActionStatus, AtomicAction, load_stage_config
from jev4mujoco.simulation.backend import MujocoBackend
from jev4mujoco.simulation.executor import AtomicActionExecutor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_stage_config(PROJECT_ROOT / "configs/executor_v1.yaml")
        self.backend = MujocoBackend(self.config)
        self.executor = AtomicActionExecutor(self.backend, self.config)

    def execute_current(self, action: AtomicAction, decision_id: int = 1):
        state = self.backend.snapshot()
        return self.executor.execute(
            ActionRequest(
                decision_id=decision_id,
                based_on_state_id=state.state_id,
                based_on_action_epoch=state.action_epoch,
                action=action,
            )
        )

    def test_all_translation_and_rotation_actions_pass(self) -> None:
        actions = (
            AtomicAction.FORWARD,
            AtomicAction.BACKWARD,
            AtomicAction.LEFT,
            AtomicAction.RIGHT,
            AtomicAction.UP,
            AtomicAction.DOWN,
            AtomicAction.ROTATE_CW,
            AtomicAction.ROTATE_CCW,
        )
        for decision_id, action in enumerate(actions, 1):
            with self.subTest(action=action.value):
                self.backend.reset()
                result = self.execute_current(action, decision_id)
                self.assertEqual(result.status, ActionStatus.COMPLETED)
                self.assertTrue(result.executed)

    def test_gripper_close_open_and_hold_pass(self) -> None:
        close_result = self.execute_current(AtomicAction.GRIPPER_CLOSE, 1)
        open_result = self.execute_current(AtomicAction.GRIPPER_OPEN, 2)
        hold_result = self.execute_current(AtomicAction.HOLD, 3)
        self.assertEqual(close_result.status, ActionStatus.COMPLETED)
        self.assertEqual(open_result.status, ActionStatus.COMPLETED)
        self.assertEqual(hold_result.status, ActionStatus.COMPLETED)

    def test_stale_state_is_rejected_without_mutation(self) -> None:
        before = self.backend.snapshot()
        result = self.executor.execute(
            ActionRequest(
                decision_id=1,
                based_on_state_id=before.state_id - 1,
                based_on_action_epoch=before.action_epoch,
                action=AtomicAction.FORWARD,
            )
        )
        after = self.backend.snapshot()
        self.assertEqual(result.status, ActionStatus.REJECTED)
        self.assertFalse(result.executed)
        self.assertEqual(before.state_id, after.state_id)
        self.assertEqual(before.action_epoch, after.action_epoch)
        self.assertEqual(before.simulation_time_s, after.simulation_time_s)

    def test_sequential_round_trip_returns_near_home(self) -> None:
        start = self.backend.snapshot()
        actions = (
            AtomicAction.FORWARD,
            AtomicAction.BACKWARD,
            AtomicAction.LEFT,
            AtomicAction.RIGHT,
            AtomicAction.UP,
            AtomicAction.DOWN,
            AtomicAction.ROTATE_CW,
            AtomicAction.ROTATE_CCW,
            AtomicAction.GRIPPER_CLOSE,
            AtomicAction.GRIPPER_OPEN,
            AtomicAction.HOLD,
        )
        for decision_id, action in enumerate(actions, 1):
            result = self.execute_current(action, decision_id)
            self.assertEqual(result.status, ActionStatus.COMPLETED)
        end = self.backend.snapshot()
        tcp_drift = np.linalg.norm(
            np.asarray(end.tcp_position_robot_base_m)
            - np.asarray(start.tcp_position_robot_base_m)
        )
        self.assertLessEqual(tcp_drift, self.config.translation_error_m)
        self.assertEqual(end.action_epoch, len(actions))

    def test_single_flight_rejects_without_mutation(self) -> None:
        before = self.backend.snapshot()
        self.executor._execution_lock.acquire()
        try:
            result = self.execute_current(AtomicAction.FORWARD)
        finally:
            self.executor._execution_lock.release()
        after = self.backend.snapshot()
        self.assertEqual(result.status, ActionStatus.REJECTED)
        self.assertEqual(result.reason, "another_action_is_in_flight")
        self.assertFalse(result.executed)
        self.assertEqual(before.state_id, after.state_id)
        self.assertEqual(before.action_epoch, after.action_epoch)
        self.assertEqual(before.simulation_time_s, after.simulation_time_s)


if __name__ == "__main__":
    unittest.main()
