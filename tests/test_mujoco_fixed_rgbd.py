from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from jev4mujoco.perception.fixed_rgbd import load_mujoco_fixed_rgbd_config_v1
from jev4mujoco.contracts.visual_scene import (
    EvidenceStatus,
    RegionSourceKind,
    Visibility,
    VisualPredicate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/pick_and_place.yaml"


def reset_to_v1_home(
    model: mujoco.MjModel, data: mujoco.MjData, keyframe_name: str
) -> None:
    keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name
    )
    if keyframe_id < 0:
        raise AssertionError(f"missing test keyframe {keyframe_name}")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)


class MujocoFixedRgbdTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_mujoco_fixed_rgbd_config_v1(CONFIG_PATH)
        self.model = mujoco.MjModel.from_xml_path(str(self.config.scene_path))
        self.data = mujoco.MjData(self.model)
        reset_to_v1_home(self.model, self.data, self.config.keyframe_name)

    def test_block_scene_loads_and_remains_finite(self) -> None:
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name
            ),
            0,
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "block_1_collision_00"
            ),
            0,
        )
        self.assertEqual(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bowl_1"),
            -1,
        )
        for _ in range(2000):
            mujoco.mj_step(self.model, self.data)
        self.assertTrue(np.isfinite(self.data.qpos).all())
        self.assertTrue(np.isfinite(self.data.qvel).all())
        self.assertTrue(np.isfinite(self.data.qacc).all())

    def test_rgbd_capture_builds_visual_scene_without_object_pose_input(self) -> None:
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)
        with self.config.make_builder(self.model) as builder:
            capture = builder.capture(
                self.data,
                scene_id="one_block_pick_and_place",
                observation_id=1,
            )

        scene = capture.scene
        self.assertTrue(scene.planning_ready)
        self.assertEqual(scene.sensor.type, "rgbd")
        self.assertEqual(scene.sensor.mounting, "fixed")
        self.assertEqual(capture.rgb.shape, (480, 640, 3))
        self.assertEqual(capture.depth_m.shape, (480, 640))
        self.assertEqual(capture.instance_segmentation.shape, (480, 640, 2))
        self.assertFalse(capture.rgb.flags.writeable)
        self.assertFalse(capture.depth_m.flags.writeable)

        entity = scene.entities[0]
        self.assertEqual(entity.track_id, "block_1")
        self.assertEqual(entity.visibility, Visibility.CLEAR)
        self.assertEqual(
            entity.measurement.method,
            "rgbd_instance_mask_depth_backprojection",
        )
        self.assertNotIn("truth", entity.measurement.method)
        self.assertEqual(
            entity.geometry.pose_6d_robot_base.status,
            EvidenceStatus.UNOBSERVABLE,
        )

        centroid = np.asarray(entity.geometry.centroid_robot_base_m.value)
        extent = np.asarray(entity.geometry.extent_m.value)
        block_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block_1"
        )
        base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.robot_base_body_name
        )
        base_rotation_world_from_base = self.data.xmat[base_body_id].reshape(3, 3)
        hidden_truth_centroid_base = (
            self.data.xpos[block_body_id] - self.data.xpos[base_body_id]
        ) @ base_rotation_world_from_base
        self.assertTrue(
            np.allclose(centroid, hidden_truth_centroid_base, atol=0.002),
            (centroid, hidden_truth_centroid_base),
        )
        self.assertTrue(np.allclose(extent, (0.03, 0.03, 0.03), atol=0.003))

        region_sources = {region.source.kind for region in scene.regions}
        self.assertEqual(region_sources, {RegionSourceKind.CALIBRATED})
        relations = {
            (relation.predicate, relation.reference_ref): relation.value
            for relation in scene.relations
        }
        self.assertTrue(relations[(VisualPredicate.ON_SURFACE, "table_surface")])
        self.assertFalse(relations[(VisualPredicate.IN_REGION, "target_region")])
        self.assertTrue(
            relations[(VisualPredicate.CLEAR_OF_REGION, "target_region")]
        )

    def test_out_of_view_block_is_not_planning_ready(self) -> None:
        free_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_1_free"
        )
        qpos_address = int(self.model.jnt_qposadr[free_joint_id])
        self.data.qpos[qpos_address] = 3.0
        mujoco.mj_forward(self.model, self.data)

        with self.config.make_builder(self.model) as builder:
            capture = builder.capture(
                self.data,
                scene_id="one_block_out_of_view",
                observation_id=2,
            )
        entity = capture.scene.entities[0]
        self.assertFalse(capture.scene.planning_ready)
        self.assertEqual(entity.visibility, Visibility.NOT_DETECTED)
        self.assertEqual(
            entity.geometry.centroid_robot_base_m.status,
            EvidenceStatus.UNOBSERVABLE,
        )
        self.assertEqual(entity.measurement.depth_valid_fraction, 0.0)

    def test_lifted_block_keeps_tracked_height_and_loses_table_support(self) -> None:
        with self.config.make_builder(self.model) as builder:
            initial = builder.capture(
                self.data,
                scene_id="one_block_tracking",
                observation_id=1,
            )
            initial_height = initial.scene.entities[0].geometry.extent_m.value[2]

            free_joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_1_free"
            )
            qpos_address = int(self.model.jnt_qposadr[free_joint_id])
            self.data.qpos[qpos_address + 2] = 0.90
            mujoco.mj_forward(self.model, self.data)
            lifted = builder.capture(
                self.data,
                scene_id="one_block_tracking",
                observation_id=2,
            )

        entity = lifted.scene.entities[0]
        self.assertIsNone(entity.support_region_ref)
        self.assertAlmostEqual(entity.geometry.extent_m.value[2], initial_height)
        block_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "block_1"
        )
        base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.robot_base_body_name
        )
        base_rotation_world_from_base = self.data.xmat[base_body_id].reshape(3, 3)
        hidden_truth_centroid_base = (
            self.data.xpos[block_body_id] - self.data.xpos[base_body_id]
        ) @ base_rotation_world_from_base
        self.assertTrue(
            np.allclose(
                entity.geometry.centroid_robot_base_m.value,
                hidden_truth_centroid_base,
                atol=0.002,
            )
        )


if __name__ == "__main__":
    unittest.main()
