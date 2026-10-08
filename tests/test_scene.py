from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from jev4mujoco import load_stage_config
from jev4mujoco.simulation.backend import MujocoBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SceneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_stage_config(PROJECT_ROOT / "configs/executor_v1.yaml")

    def test_keyframe_reset_is_exactly_reproducible(self) -> None:
        backend = MujocoBackend(self.config)
        expected = backend.snapshot()
        backend.step(50)
        actual = backend.reset()
        self.assertEqual(actual.joint_positions_rad, expected.joint_positions_rad)
        self.assertEqual(actual.gripper_joint_m, expected.gripper_joint_m)
        self.assertEqual(actual.bowl_position_world_m, expected.bowl_position_world_m)

    def test_episode_object_position_override_is_reproducible_on_reset(self) -> None:
        requested = (-0.16, -0.03, 0.765)
        backend = MujocoBackend(self.config, requested)
        initial = backend.snapshot()
        self.assertTrue(np.allclose(initial.bowl_position_world_m, requested, atol=1e-12))

        backend.step(50)
        reset = backend.reset()
        self.assertTrue(np.allclose(reset.bowl_position_world_m, requested, atol=1e-12))

    def test_two_thousand_steps_remain_finite(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(self.config.scene_path))
        data = mujoco.MjData(model)
        keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, self.config.keyframe_name
        )
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        for _ in range(2000):
            mujoco.mj_step(model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.qvel).all())
        self.assertTrue(np.isfinite(data.qacc).all())


if __name__ == "__main__":
    unittest.main()
