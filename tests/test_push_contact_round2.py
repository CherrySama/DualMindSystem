from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from jev4mujoco.contracts.actions import AtomicAction
from jev4mujoco.contracts.local_intent import LocalIntentChoiceV1
from jev4mujoco.contracts.tool_geometry import ToolGeometryV1, ToolPartGeometryV1
from jev4mujoco.contracts.visual_scene import Visibility, VisualPredicate
from jev4mujoco.experiments.scenarios import BlockSurfacePushRuntimeV1
from jev4mujoco.perception.fixed_rgbd import load_mujoco_fixed_rgbd_config_v1
from jev4mujoco.planning.capabilities import CapabilityKind, VerificationOutcome
from jev4mujoco.policies.jev_context import build_local_intent_motor_context_v1, _contact_context
from jev4mujoco.runtime.local_intent import LocalIntentRuntimeV1, load_local_intent_config_v1
from jev4mujoco.runtime.tool_geometry import CalibratedFingerToolV1, contact_geometry_v1, observed_contact_sides_v1, _convex_closest_points
from tests import test_shared_subtask_context as fixtures
from tests.test_shared_subtask_context import phase_state, payload
from tests.test_visual_scene_v1 import measured

ROOT = Path(__file__).resolve().parents[1]
FACES = ((0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
    (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
    (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3))


def box_tool(y=(-0.195, -0.165), z=(0.005, 0.025), x=(0.29, 0.31)):
    vertices = tuple(product(x, y, z))
    return ToolGeometryV1((ToolPartGeometryV1("test_finger", vertices, FACES),), "test_calibration")


class PushContactRound2Test(unittest.TestCase):
    def setUp(self):
        self.assessor = fixtures.SharedSubtaskContextTest()
        self.assessor.setUp()
        self.state, self.program, self.execution = phase_state(CapabilityKind.SURFACE_PUSH, "establish_contact")
        self.state = replace(self.state, robot=replace(self.state.robot, tool_geometry=box_tool()))
        self.config = load_local_intent_config_v1(ROOT / "configs/local_intent_v1.yaml")
        self.runtime = LocalIntentRuntimeV1(self.config)
        self.side = observed_contact_sides_v1(self.state)[1]

    def attach(self, state=None):
        state = self.state if state is None else state
        state = self.assessor.assessed(replace(state, local_intent=None, phase_feedback=None), self.program, self.execution)
        return replace(state, local_intent=self.runtime.context(state))

    def select(self, state, name="contact_edge_1"):
        return self.runtime.accept(LocalIntentChoiceV1(name, state.state_id, state.action_epoch,
            state.skill.skill_instance_id, state.skill.phase), state)

    def test_normal_translation_reduces_gap_while_tangent_translation_does_not(self):
        initial = contact_geometry_v1(self.state, self.side)
        forward = contact_geometry_v1(self.state, self.side, (0.01, 0, 0))
        left = contact_geometry_v1(self.state, self.side, (0, 0.01, 0))
        self.assertAlmostEqual(initial["local_error_m"], 0.035)
        self.assertAlmostEqual(forward["local_error_m"], 0.035)
        self.assertAlmostEqual(left["local_error_m"], 0.025)

    def test_finite_side_miss_has_distance_and_correction_despite_unknown_normal_gap(self):
        geometry = contact_geometry_v1(self.state, self.side, (0.10, 0, 0))
        self.assertIsNone(geometry["parts"][0]["normal_gap_m"])
        self.assertGreater(geometry["parts"][0]["tangent_deficit_m"], 0)
        self.assertGreater(geometry["local_error_m"], 0.035)
        self.assertLess(geometry["correction_robot_base_m"][0], 0)
        self.assertGreater(geometry["correction_robot_base_m"][1], 0)

    def test_tool_behind_opposite_side_is_not_a_zero_gap(self):
        opposite = observed_contact_sides_v1(self.state)[3]
        geometry = contact_geometry_v1(self.state, opposite)
        self.assertLess(geometry["parts"][0]["normal_gap_m"], 0)
        self.assertGreater(geometry["local_error_m"], 0)
        self.assertEqual(geometry["parts"][0]["approach_side_status"], "behind")

    def test_above_subject_does_not_use_upper_tool_width_as_contact(self):
        state = replace(self.state, robot=replace(self.state.robot, tool_geometry=box_tool(z=(0.08, 0.10))))
        geometry = contact_geometry_v1(state, self.side)
        self.assertGreater(geometry["local_error_m"], 0.035)
        self.assertLess(geometry["correction_robot_base_m"][2], 0)
        self.assertGreater(geometry["parts"][0]["height_deficit_m"], 0)
        self.assertFalse(geometry["parts"][0]["finite_patch_overlap"])

    def test_closest_distance_matches_analytic_boxes_with_witnesses(self):
        patch = np.array(tuple(product((0.0,), (-0.02, 0.02), (-0.01, 0.01))))
        rng = np.random.default_rng(104)
        for center in rng.uniform(-0.10, 0.10, size=(50, 3)):
            lower, upper = center - (0.01, 0.015, 0.005), center + (0.01, 0.015, 0.005)
            vertices = np.array(tuple(product(*zip(lower, upper))))
            distance, tool_point, side_point = _convex_closest_points(vertices, patch)
            deficits = np.maximum(np.maximum(lower - patch.max(axis=0), patch.min(axis=0) - upper), 0)
            self.assertAlmostEqual(distance, np.linalg.norm(deficits), places=8)
            self.assertAlmostEqual(distance, np.linalg.norm(side_point - tool_point), places=8)
            self.assertTrue(np.all(tool_point >= lower - 1e-8) and np.all(tool_point <= upper + 1e-8))
            self.assertTrue(np.all(side_point >= patch.min(axis=0) - 1e-8)
                and np.all(side_point <= patch.max(axis=0) + 1e-8))

    def test_rotated_thin_tool_uses_hull_not_independent_axis_overlap(self):
        # A diagonal hull misses the finite square despite overlapping both axis ranges.
        vertices = np.array([(x, y, z) for x, y in ((0, 0.03), (0.03, 0), (0.033, 0.003), (0.003, 0.033))
            for z in (-0.005, 0.005)])
        patch = np.array(tuple(product((0.0, 0.01), (0.0, 0.01), (0.0,))))
        distance, _, _ = _convex_closest_points(vertices, patch)
        self.assertAlmostEqual(distance, 0.01 / np.sqrt(2), places=8)

    def test_polygon_winding_preserves_outward_normal(self):
        entity = self.state.visual_scene.entities[0]
        reversed_entity = replace(entity, geometry=replace(entity.geometry,
            footprint_xy_robot_base_m=measured(tuple(reversed(entity.geometry.footprint_xy_robot_base_m.value)))))
        state = replace(self.state, visual_scene=replace(self.state.visual_scene, entities=(reversed_entity,)))
        counterpart = next(side for side in observed_contact_sides_v1(state)
            if side.outward_normal_xy == self.side.outward_normal_xy)
        self.assertAlmostEqual(contact_geometry_v1(state, counterpart)["local_error_m"], 0.035)

    def test_zero_geometric_gap_does_not_pass_contact_or_finish_intent(self):
        state = replace(self.state, robot=replace(self.state.robot, tool_geometry=box_tool(y=(-0.16, -0.13))))
        state = self.attach(state)
        self.select(state)
        state = replace(state, local_intent=self.runtime.context(state))
        self.assertEqual(state.local_intent.execution.metric_m, 0)
        self.assertEqual(state.local_intent.execution.status, "active")
        result = self.assessor.verifier.verify_active_phase(self.program, self.execution, state)
        self.assertIsNot(result.outcome, VerificationOutcome.PASSED)

    def test_unreached_selected_side_does_not_restrict_next_phase(self):
        before, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        before = replace(before, robot=replace(before.robot, tool_geometry=box_tool()))
        before = self.assessor.assessed(before, program, execution)
        before = replace(before, local_intent=self.runtime.context(before))
        self.select(before, "approach_edge_1")
        contact = self.attach()
        self.assertEqual({c.choice_id for c in contact.local_intent.candidates},
            {f"contact_edge_{i}" for i in range(4)})
        self.assertIsNone(contact.local_intent.handoff)

    def test_single_and_two_stage_inputs_share_geometry_and_exclude_raw_mesh(self):
        state = self.attach()
        full = payload(state)
        self.assertEqual(len(full["push_contact_geometry"]), 4)
        candidate = next(c for c in state.local_intent.candidates if c.choice_id == "contact_edge_1")
        motor = build_local_intent_motor_context_v1(state, candidate, full)
        expected = next(c for c in full["push_contact_geometry"] if c["geometry"]["side"]["edge_index"] == 1)
        self.assertEqual(motor["contact_geometry"], expected["geometry"])
        evaluations = {e["action"]: e for e in motor["action_evaluations"]}
        self.assertAlmostEqual(evaluations["left"]["projected_progress_m"], 0.01)
        self.assertAlmostEqual(evaluations["forward"]["projected_progress_m"], 0)
        self.assertFalse(evaluations["left"]["is_observed"])
        self.assertNotIn("vertices_robot_base_m", json.dumps(motor))

    def test_motor_can_predict_progress_before_finite_patch_overlap(self):
        state = replace(self.state, robot=replace(self.state.robot,
            tool_geometry=box_tool(x=(0.39, 0.41))))
        state = self.attach(state)
        candidate = next(c for c in state.local_intent.candidates if c.choice_id == "contact_edge_1")
        motor = build_local_intent_motor_context_v1(state, candidate, payload(state))
        self.assertIsNone(motor["contact_geometry"]["parts"][0]["normal_gap_m"])
        self.assertIsNotNone(motor["selected_local_intent"]["correction_robot_base_m"])
        evaluations = {item["action"]: item for item in motor["action_evaluations"]}
        self.assertGreater(evaluations["backward"]["projected_progress_m"], 0)
        self.assertGreater(evaluations["left"]["projected_progress_m"], 0)
        self.assertLess(evaluations["forward"]["projected_progress_m"], 0)

    def test_compact_predictions_preserve_four_sides_and_action_comparisons(self):
        state = self.attach()
        full = payload(state)
        sides = observed_contact_sides_v1(state)
        self.assertEqual(len(full["push_contact_geometry"]), len(sides))
        for side, context in zip(sides, full["push_contact_geometry"]):
            current = contact_geometry_v1(state, side)
            self.assertEqual(context["geometry"], current)
            self.assertFalse(current["is_contact_confirmation"])
            self.assertFalse(current["path_verified"])
            self.assertEqual({e["action"] for e in context["action_evaluations"]}, set(state.skill.allowed_actions))
            for evaluation in context["action_evaluations"]:
                predicted = contact_geometry_v1(state, side, evaluation["expected_tcp_delta_robot_base_m"])
                self.assertEqual(evaluation["projected_local_error_m"], predicted["local_error_m"])
                self.assertEqual(evaluation["projected_progress_m"], current["local_error_m"] - predicted["local_error_m"])
                self.assertFalse(evaluation["is_observed"])
                self.assertEqual(evaluation["source"], "bounded_command_geometry")
                self.assertIn("ignores dynamics and path collisions", evaluation["assumptions"])
                summary = evaluation["projected_contact_summary"]
                self.assertEqual(summary["observable"], predicted["observable"])
                self.assertEqual(len(summary["parts"]), len(predicted["parts"]))
                for actual, expected in zip(summary["parts"], predicted["parts"]):
                    self.assertEqual(set(actual), {"part_id", "normal_gap_m", "tangent_deficit_m",
                        "height_deficit_m", "finite_patch_overlap", "local_error_m", "approach_side_status"})
                    self.assertEqual(actual, {key: expected[key] for key in actual})
                    self.assertNotIn("closest_tool_point_robot_base_m", actual)
                self.assertNotIn("projected_contact_geometry", evaluation)
                self.assertNotIn("side", summary)
        candidate = next(c for c in state.local_intent.candidates if c.choice_id == "contact_edge_1")
        motor = build_local_intent_motor_context_v1(state, candidate, full)
        self.assertEqual(motor["action_evaluations"], full["push_contact_geometry"][1]["action_evaluations"])
        self.assertEqual(motor["phase"]["conditions"], full["phase"]["conditions"])
        self.assertEqual(motor["phase"]["maintain_constraints"], full["phase"]["maintain_constraints"])

    def test_compact_predictions_keep_distinct_parts_and_negative_gap(self):
        outside = box_tool().parts[0]
        behind = replace(box_tool(y=(-0.11, -0.09)).parts[0], part_id="second_finger")
        state = replace(self.state, robot=replace(self.state.robot,
            tool_geometry=ToolGeometryV1((outside, behind), "test_calibration")))
        context = _contact_context(state, self.side, ("hold",), 0.01)
        parts = context["action_evaluations"][0]["projected_contact_summary"]["parts"]
        self.assertEqual([p["part_id"] for p in parts], ["test_finger", "second_finger"])
        self.assertGreater(parts[0]["normal_gap_m"], 0)
        self.assertLess(parts[1]["normal_gap_m"], 0)
        self.assertEqual(parts[1]["approach_side_status"], "behind")

    def test_compact_predictions_keep_patch_miss_and_height_deficit(self):
        state = replace(self.state, robot=replace(self.state.robot,
            tool_geometry=box_tool(x=(0.39, 0.41), z=(0.08, 0.10))))
        context = _contact_context(state, self.side, ("hold",), 0.01)
        part = context["action_evaluations"][0]["projected_contact_summary"]["parts"][0]
        self.assertIsNone(part["normal_gap_m"])
        self.assertFalse(part["finite_patch_overlap"])
        self.assertGreater(part["tangent_deficit_m"], 0)
        self.assertGreater(part["height_deficit_m"], 0)
        self.assertEqual(part["approach_side_status"], "no_finite_patch_overlap")

    def test_compact_predictions_preserve_unobservable_and_failed_distance(self):
        entity = replace(self.state.visual_scene.entities[0], visibility=Visibility.PARTIAL)
        state = replace(self.state, visual_scene=replace(self.state.visual_scene, entities=(entity,)))
        context = _contact_context(state, self.side, ("hold",), 0.01)
        evaluation = context["action_evaluations"][0]
        self.assertFalse(evaluation["projected_contact_summary"]["observable"])
        self.assertTrue(evaluation["projected_contact_summary"]["reason"])
        self.assertEqual(evaluation["projected_contact_summary"]["parts"], [])
        self.assertIsNone(evaluation["projected_local_error_m"])
        self.assertIsNone(evaluation["projected_progress_m"])
        with patch("jev4mujoco.runtime.tool_geometry._convex_closest_points", return_value=None):
            context = _contact_context(self.state, self.side, ("hold",), 0.01)
        evaluation = context["action_evaluations"][0]
        self.assertTrue(evaluation["projected_contact_summary"]["observable"])
        self.assertIsNone(evaluation["projected_contact_summary"]["parts"][0]["local_error_m"])
        self.assertIsNone(evaluation["projected_local_error_m"])
        self.assertIsNone(evaluation["projected_progress_m"])

    def test_partial_geometry_invalidates_contact_and_is_not_zero_progress(self):
        state = self.attach()
        self.select(state)
        entity = replace(state.visual_scene.entities[0], visibility=Visibility.PARTIAL)
        state = self.attach(replace(state, visual_scene=replace(state.visual_scene, entities=(entity,))))
        self.assertEqual(state.local_intent.execution.status, "invalidated")
        self.assertIsNone(state.local_intent.execution.metric_m)
        self.assertFalse(state.local_intent.candidates)
        self.assertIsNone(state.local_intent.retained_approach)
        self.assertFalse(contact_geometry_v1(state, self.side)["observable"])

    def test_repeated_tangent_actions_trigger_existing_nonprogress_replanning(self):
        state = self.attach()
        self.select(state)
        for _ in range(self.config.maximum_actions_without_new_best):
            state = replace(state, state_id=state.state_id + 1, action_epoch=state.action_epoch + 1,
                phase_feedback=None, local_intent=None,
                recent_action=replace(state.recent_action, start_action_epoch=state.action_epoch,
                    end_action_epoch=state.action_epoch + 1))
            state = self.attach(state)
        self.assertEqual(state.local_intent.execution.status, "invalidated")
        self.assertEqual(state.local_intent.replan_count, 1)

    def test_unknown_right_finger_feedback_does_not_assume_symmetry(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / "scenes/push_aside.xml"))
        tool = CalibratedFingerToolV1.from_model(model)
        self.assertIsNone(tool.observe((0, 0, 0), (1, 0, 0, 0), 0.02, None))

    def test_partial_subject_movement_is_rejected_but_hold_remains_available(self):
        with BlockSurfacePushRuntimeV1(ROOT, desired_inside_target=False) as runtime:
            state = replace(self.state, visual_scene=replace(self.state.visual_scene,
                entities=(replace(self.state.visual_scene.entities[0], visibility=Visibility.PARTIAL),)))
            review = runtime._review_push_geometry_decision(SimpleNamespace(action=AtomicAction.LEFT, decision_id=1), state)
            self.assertEqual(review.status, "rejected")
            self.assertIsNone(runtime._review_push_geometry_decision(SimpleNamespace(action=AtomicAction.HOLD), state))


class RobotAndVisionRound2Test(unittest.TestCase):
    def test_calibrated_hulls_match_robot_geoms_with_independent_fingers(self):
        with BlockSurfacePushRuntimeV1(ROOT, desired_inside_target=False) as runtime:
            model, data = runtime.backend.model, runtime.backend._data
            for name, value in (("gripper_L_joint", 0.02), ("gripper_R_joint", -0.011)):
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[model.jnt_qposadr[joint]] = value
            mujoco.mj_forward(model, data)
            raw = runtime.backend.snapshot()
            geometry = runtime._tool_calibration.observe(raw.tcp_position_robot_base_m,
                raw.tcp_orientation_robot_base_wxyz, raw.gripper_joint_m, raw.right_gripper_joint_m)
            self.assertAlmostEqual(raw.right_gripper_joint_m, -0.011)
            base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panthera")
            for part in geometry.parts:
                geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, part.part_id)
                mesh = model.geom_dataid[geom]
                vertices = np.asarray(model.mesh_vert[model.mesh_vertadr[mesh]:
                    model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]], dtype=float)
                world = np.einsum("ij,kj->ki", data.geom_xmat[geom].reshape(3, 3), vertices) + data.geom_xpos[geom]
                expected = np.einsum("ij,ki->kj", data.xmat[base].reshape(3, 3), world - data.xpos[base])
                actual = np.asarray(part.vertices_robot_base_m)
                np.testing.assert_allclose(actual.min(axis=0), expected.min(axis=0), atol=1e-8)
                np.testing.assert_allclose(actual.max(axis=0), expected.max(axis=0), atol=1e-8)

    def test_visible_fragment_is_partial_and_reappearance_restores_fresh_geometry(self):
        config = load_mujoco_fixed_rgbd_config_v1(ROOT / "configs/push_aside.yaml")
        model = mujoco.MjModel.from_xml_path(str(config.scene_path))
        with config.make_builder(model) as builder:
            spec = config.entity_specs[0]
            region = builder._build_region(config.calibrated_regions[0])
            def observe(width, x=0.35):
                points = np.asarray(tuple(product((x - 0.015, x + 0.015),
                    (0.18 - width / 2, 0.18 + width / 2), (0.03,))))
                return builder._build_entity(spec, mask_pixel_count=len(points),
                    points_robot_base_m=points, support_region=region, xy_span_uncertainty_m=0.004)
            self.assertIs(observe(0.03).visibility, Visibility.CLEAR)
            fragment = observe(0.0055)
            self.assertIs(fragment.visibility, Visibility.PARTIAL)
            self.assertIsNone(fragment.geometry.footprint_xy_robot_base_m.value)
            self.assertIsNone(fragment.geometry.centroid_robot_base_m.value)
            self.assertFalse(builder._derive_relations((fragment,), (region,)))
            restored = observe(0.03, x=0.40)
            self.assertIs(restored.visibility, Visibility.CLEAR)
            self.assertAlmostEqual(restored.geometry.centroid_robot_base_m.value[0], 0.40)

    def test_raster_uncertainty_does_not_reject_small_width_changes(self):
        config = load_mujoco_fixed_rgbd_config_v1(ROOT / "configs/push_aside.yaml")
        model = mujoco.MjModel.from_xml_path(str(config.scene_path))
        with config.make_builder(model) as builder:
            spec = config.entity_specs[0]
            region = builder._build_region(config.calibrated_regions[0])
            for width in (0.03, 0.027):
                points = np.asarray(tuple(product((0.335, 0.365), (0.18 - width/2, 0.18 + width/2), (0.03,))))
                entity = builder._build_entity(spec, mask_pixel_count=4, points_robot_base_m=points,
                    support_region=region, xy_span_uncertainty_m=0.004)
                self.assertIs(entity.visibility, Visibility.CLEAR)


if __name__ == "__main__":
    unittest.main()
