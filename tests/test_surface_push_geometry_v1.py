from __future__ import annotations

import unittest

from jev4mujoco.perception.surface_geometry import (
    footprint_clear_of_region_v1,
    polygon_clearance_v1,
    polygons_overlap_v1,
)


class SurfacePushGeometryV1Test(unittest.TestCase):
    def test_partial_exit_still_overlaps_and_is_not_clear(self) -> None:
        region = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        footprint = ((0.8, 0.2), (1.2, 0.2), (1.2, 0.8), (0.8, 0.8))

        self.assertTrue(polygons_overlap_v1(footprint, region))
        self.assertEqual(polygon_clearance_v1(footprint, region), 0.0)
        self.assertFalse(footprint_clear_of_region_v1(footprint, region, 0.05))

    def test_complete_exit_must_also_reach_configured_clearance(self) -> None:
        region = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        too_close = ((1.01, 0.2), (1.21, 0.2), (1.21, 0.8), (1.01, 0.8))
        clear = ((1.06, 0.2), (1.26, 0.2), (1.26, 0.8), (1.06, 0.8))

        self.assertFalse(polygons_overlap_v1(too_close, region))
        self.assertAlmostEqual(polygon_clearance_v1(too_close, region), 0.01)
        self.assertFalse(footprint_clear_of_region_v1(too_close, region, 0.05))
        self.assertTrue(footprint_clear_of_region_v1(clear, region, 0.05))


if __name__ == "__main__":
    unittest.main()
