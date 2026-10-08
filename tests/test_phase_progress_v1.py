from __future__ import annotations

import unittest
from math import hypot

from jev4mujoco.contracts.actions import AtomicAction
from jev4mujoco.runtime.progress import PickXYErrorV1, projected_pick_xy_distance_v1


class PickXYProgressV1Test(unittest.TestCase):
    def test_failed_first_request_rejects_left_and_accepts_right(self) -> None:
        # First request from attempt_002/jev_request_decision_001.json.
        delta_x, delta_y = -0.026600017798650344, -0.179866571714537
        before = PickXYErrorV1(delta_x, delta_y, hypot(delta_x, delta_y))
        step = 0.01

        left = projected_pick_xy_distance_v1(before, AtomicAction.LEFT, step)
        right = projected_pick_xy_distance_v1(before, AtomicAction.RIGHT, step)

        self.assertGreater(left, before.distance_m)
        self.assertLess(right, before.distance_m)


if __name__ == "__main__":
    unittest.main()
