from __future__ import annotations

import json
import unittest
from dataclasses import replace

from jev4mujoco.policies.jev_context import build_jev_decision_context_v1
from tests.test_state_snapshot_v2 import snapshot


class JevDecisionContextV1Test(unittest.TestCase):
    def test_context_keeps_only_current_decision_inputs(self) -> None:
        state = snapshot()
        state = replace(
            state,
            skill=replace(
                state.skill,
                phase="align_subject_xy",
                allowed_intents=("align_subject_xy",),
                allowed_actions=(
                    "forward",
                    "backward",
                    "left",
                    "right",
                    "hold",
                ),
            ),
        )

        payload = build_jev_decision_context_v1(
            state, translation_step_m=0.01, grasp_xy_tolerance_m=0.008
        ).to_dict()
        encoded = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["schema"], "JEVDecisionContextV1")
        self.assertEqual(payload["subject"]["ref"], "track_001")
        self.assertIsNone(payload["target"])
        delta = payload["interaction"]["goal_delta_robot_base_m"]
        self.assertAlmostEqual(delta[0], -0.05)
        self.assertAlmostEqual(delta[1], -0.10)
        self.assertEqual(delta[2], 0.0)
        self.assertEqual(payload["interaction"]["active_axes"], ["x", "y"])
        self.assertEqual(payload["interaction"]["completion_tolerance_m"], 0.008)
        self.assertEqual(payload["handle"]["translation_step_m"], 0.01)
        self.assertNotIn("up", payload["handle"]["available_actions"])
        self.assertNotIn("up", payload["handle"]["directions_robot_base"])
        self.assertNotIn("visual_scene", payload)
        self.assertNotIn("joint_positions_rad", encoded)
        self.assertNotIn("left_region", encoded)

    def test_translation_step_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "translation_step_m"):
            build_jev_decision_context_v1(
                snapshot(), translation_step_m=0.0, grasp_xy_tolerance_m=0.008
            )


if __name__ == "__main__":
    unittest.main()
