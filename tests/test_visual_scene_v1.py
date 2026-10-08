from __future__ import annotations

import json
import unittest

from jev4mujoco.contracts.visual_scene import (
    EntityGeometryV1,
    EntityMeasurementV1,
    EntitySemanticV1,
    EntityType,
    EntityV1,
    EvidenceStatus,
    EvidenceValueV1,
    ObservationScopeMode,
    ObservationScopeV1,
    RegionGeometryV1,
    RegionSourceKind,
    RegionSourceV1,
    RegionType,
    RegionV1,
    SensorV1,
    Visibility,
    VisualDataRefsV1,
    VisualPredicate,
    VisualRelationV1,
    VisualSceneV1,
)


def measured(value):
    return EvidenceValueV1(value, EvidenceStatus.MEASURED, 1.0)


def derived(value):
    return EvidenceValueV1(value, EvidenceStatus.DERIVED, 0.99)


def scene() -> VisualSceneV1:
    table = RegionV1(
        region_id="table_surface",
        region_type=RegionType.SURFACE,
        semantic_label="table_surface",
        owner_entity_ref=None,
        geometry=RegionGeometryV1(
            boundary_xy_robot_base_m=measured(
                ((0.0, -0.4), (0.7, -0.4), (0.7, 0.4), (0.0, 0.4))
            ),
            surface_height_robot_base_m=measured(0.0),
            surface_normal_robot_base=measured((0.0, 0.0, 1.0)),
        ),
        source=RegionSourceV1(RegionSourceKind.CALIBRATED, 1.0),
    )
    target = RegionV1(
        region_id="left_region",
        region_type=RegionType.REGION_2D,
        semantic_label="left_sorting_region",
        owner_entity_ref=None,
        geometry=RegionGeometryV1(
            boundary_xy_robot_base_m=measured(
                ((0.2, 0.05), (0.35, 0.05), (0.35, 0.20), (0.2, 0.20))
            ),
            surface_height_robot_base_m=measured(0.0),
            surface_normal_robot_base=measured((0.0, 0.0, 1.0)),
        ),
        source=RegionSourceV1(RegionSourceKind.CALIBRATED, 1.0),
    )
    block = EntityV1(
        track_id="track_001",
        entity_type=EntityType.MOVABLE_OBJECT,
        semantic=EntitySemanticV1(
            category=measured("block"),
            color=measured("red"),
        ),
        visibility=Visibility.CLEAR,
        geometry=EntityGeometryV1(
            centroid_robot_base_m=measured((0.30, -0.10, 0.03)),
            top_center_robot_base_m=derived((0.30, -0.10, 0.06)),
            extent_m=derived((0.06, 0.06, 0.06)),
            footprint_xy_robot_base_m=derived(
                ((0.27, -0.13), (0.33, -0.13), (0.33, -0.07), (0.27, -0.07))
            ),
            surface_normal_robot_base=derived((0.0, 0.0, 1.0)),
            pose_6d_robot_base=EvidenceValueV1(
                None, EvidenceStatus.UNOBSERVABLE, 0.0
            ),
        ),
        support_region_ref="table_surface",
        measurement=EntityMeasurementV1(
            method="rgbd_instance_mask_points",
            depth_valid_fraction=1.0,
        ),
    )
    return VisualSceneV1(
        scene_id="scene_000001",
        observation_id=1,
        capture_timestamp_s=0.0,
        publish_timestamp_s=0.02,
        sensor=SensorV1(
            sensor_id="fixed_rgbd_1",
            type="rgbd",
            mounting="fixed",
            optical_frame="fixed_rgbd_optical",
            calibration_id="fixed_rgbd_calibration_v1",
        ),
        observation_scope=ObservationScopeV1(ObservationScopeMode.FULL_WORKSPACE),
        planning_ready=True,
        entities=(block,),
        regions=(table, target),
        relations=(
            VisualRelationV1(
                predicate=VisualPredicate.ON_SURFACE,
                subject_ref="track_001",
                reference_ref="table_surface",
                value=True,
                status=EvidenceStatus.DERIVED,
                confidence=0.99,
            ),
        ),
        data_refs=VisualDataRefsV1(
            rgb="observation://1/rgb",
            depth="observation://1/depth",
            instance_masks="observation://1/masks",
            point_cloud="observation://1/points",
        ),
    )


class VisualSceneV1Test(unittest.TestCase):
    def test_ideal_fixed_rgbd_scene_is_json_serializable(self) -> None:
        payload = scene().to_dict()

        self.assertEqual(payload["schema"], "VisualSceneV1")
        self.assertEqual(payload["coordinate_frame"], "robot_base")
        self.assertEqual(payload["sensor"]["mounting"], "fixed")
        self.assertEqual(payload["entities"][0]["semantic"]["color"]["value"], "red")
        self.assertEqual(payload["regions"][1]["region_id"], "left_region")
        json.dumps(payload, sort_keys=True)

    def test_unobservable_evidence_cannot_carry_a_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "unobservable evidence"):
            EvidenceValueV1((0.0, 0.0, 0.0), EvidenceStatus.UNOBSERVABLE, 0.0)

    def test_planning_ready_requires_full_workspace(self) -> None:
        original = scene()
        with self.assertRaisesRegex(ValueError, "planning_ready"):
            VisualSceneV1(
                scene_id=original.scene_id,
                observation_id=original.observation_id,
                capture_timestamp_s=original.capture_timestamp_s,
                publish_timestamp_s=original.publish_timestamp_s,
                sensor=original.sensor,
                observation_scope=ObservationScopeV1(
                    ObservationScopeMode.ATTENTION_SUBSET,
                    ("track_001",),
                ),
                planning_ready=True,
                entities=original.entities,
                regions=original.regions,
                relations=original.relations,
                data_refs=original.data_refs,
            )

    def test_references_must_resolve_inside_the_scene(self) -> None:
        original = scene()
        bad_entity = EntityV1(
            track_id=original.entities[0].track_id,
            entity_type=original.entities[0].entity_type,
            semantic=original.entities[0].semantic,
            visibility=original.entities[0].visibility,
            geometry=original.entities[0].geometry,
            support_region_ref="missing_surface",
            measurement=original.entities[0].measurement,
        )
        with self.assertRaisesRegex(ValueError, "missing support region"):
            VisualSceneV1(
                scene_id=original.scene_id,
                observation_id=original.observation_id,
                capture_timestamp_s=original.capture_timestamp_s,
                publish_timestamp_s=original.publish_timestamp_s,
                sensor=original.sensor,
                observation_scope=original.observation_scope,
                planning_ready=original.planning_ready,
                entities=(bad_entity,),
                regions=original.regions,
                relations=original.relations,
                data_refs=original.data_refs,
            )


if __name__ == "__main__":
    unittest.main()
