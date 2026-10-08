from __future__ import annotations

import unittest
from dataclasses import replace

from jev4mujoco.perception.relations import (
    VisualRelationConfigV1,
    derive_entity_relations_v1,
)
from jev4mujoco.contracts.visual_scene import (
    EvidenceStatus,
    EvidenceValueV1,
    VisualPredicate,
)
from tests.test_visual_scene_v1 import scene


def derived(value):
    return EvidenceValueV1(value, EvidenceStatus.DERIVED, 1.0)


def entity_at(track_id: str, xyz: tuple[float, float, float]):
    original = scene().entities[0]
    half = 0.025
    x, y, z = xyz
    geometry = replace(
        original.geometry,
        centroid_robot_base_m=derived(xyz),
        top_center_robot_base_m=derived((x, y, z + half)),
        extent_m=derived((2 * half, 2 * half, 2 * half)),
        footprint_xy_robot_base_m=derived(
            (
                (x - half, y - half),
                (x + half, y - half),
                (x + half, y + half),
                (x - half, y + half),
            )
        ),
    )
    return replace(original, track_id=track_id, geometry=geometry)


class VisualRelationsV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = VisualRelationConfigV1(
            directional_gap_m=0.02,
            near_maximum_clearance_m=0.08,
            separated_minimum_clearance_m=0.12,
            support_height_tolerance_m=0.012,
            support_minimum_overlap_fraction=0.5,
        )

    def test_robot_base_direction_and_distance_relations(self) -> None:
        reference = entity_at("reference", (0.30, 0.00, 0.025))
        left_near = entity_at("left_near", (0.30, 0.10, 0.025))
        front_far = entity_at("front_far", (0.50, 0.00, 0.025))
        relations = derive_entity_relations_v1(
            (reference, left_near, front_far), self.config
        )
        values = {
            (item.subject_ref, item.predicate, item.reference_ref): item.value
            for item in relations
        }

        self.assertTrue(
            values[("left_near", VisualPredicate.LEFT_OF, "reference")]
        )
        self.assertTrue(values[("left_near", VisualPredicate.NEAR, "reference")])
        self.assertTrue(
            values[("front_far", VisualPredicate.IN_FRONT_OF, "reference")]
        )
        self.assertTrue(
            values[("front_far", VisualPredicate.SEPARATED, "reference")]
        )

    def test_object_on_object_requires_height_and_support_overlap(self) -> None:
        base = entity_at("base", (0.30, 0.00, 0.025))
        stacked = entity_at("stacked", (0.30, 0.00, 0.075))
        offset = entity_at("offset", (0.36, 0.00, 0.075))
        relations = derive_entity_relations_v1((base, stacked, offset), self.config)
        values = {
            (item.subject_ref, item.predicate, item.reference_ref): item.value
            for item in relations
        }

        self.assertTrue(
            values[("stacked", VisualPredicate.ON_SURFACE, "base")]
        )
        self.assertFalse(
            values[("offset", VisualPredicate.ON_SURFACE, "base")]
        )


if __name__ == "__main__":
    unittest.main()
