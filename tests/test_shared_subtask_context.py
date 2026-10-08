from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from jev4mujoco.contracts.state_snapshot import EffectorContactMode, SkillGoalV1
from jev4mujoco.contracts.visual_scene import EvidenceStatus, EvidenceValueV1, RegionType, VisualPredicate
from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1, CapabilityGoalV1, CapabilityInstanceV1, CapabilityKind,
    CapabilityProgramV1, VerificationOutcome,
)
from jev4mujoco.planning.phase_definitions import bound_subtask_definition_v1
from jev4mujoco.policies.jev_context import build_jev_decision_context_v1
from jev4mujoco.runtime.transitions import initial_capability_execution_state_v1, start_capability_program_v1, active_skill_state_v2
from jev4mujoco.runtime.verifier import RuntimeVerifierV1
from jev4mujoco.runtime.verifier_config import load_runtime_verifier_config_v1
from tests.test_state_snapshot_v2 import snapshot
from tests.test_visual_scene_v1 import measured
from tests.test_capability_jev_policy_v1 import response


ROOT = Path(__file__).resolve().parents[1]


def phase_state(kind, phase, predicate=VisualPredicate.IN_REGION):
    original = snapshot()
    original = replace(original,
        visual_scene=replace(original.visual_scene,
            capture_timestamp_s=original.timestamp_s, publish_timestamp_s=original.timestamp_s),
        sources=replace(original.sources, visual_capture_timestamp_s=original.timestamp_s),
        freshness=replace(original.freshness, visual_age_s=0.0),
        # 测试夹具显式声明模拟证据来源，不能用默认 unknown 当已核验观测。
        physical=replace(original.physical, evidence=replace(original.physical.evidence,
            contact_source="fixture_contact_sensor", grasp_source="fixture_grasp_estimate",
            support_source="fixture_support_estimate")))
    template = CAPABILITY_TEMPLATES_V1[kind]
    if kind is CapabilityKind.PRESS and predicate is VisualPredicate.IN_REGION:
        predicate = VisualPredicate.PRESSED
    goal = None if kind is CapabilityKind.PICK else CapabilityGoalV1(predicate, "left_region", True)
    instance = CapabilityInstanceV1("shared_subject", template.template_id, kind,
        "track_001", "table_surface", None if goal is None else "left_region", 0, goal)
    program = CapabilityProgramV1("shared_program", original.task.task_id, original.visual_scene.scene_id, (instance,))
    execution = start_capability_program_v1(program, initial_capability_execution_state_v1(program))
    step = replace(execution.steps[0], phase_index=next(
        index for index, item in enumerate(template.phases) if item.phase_id == phase))
    execution = replace(execution, steps=(step,))
    return replace(original, skill=active_skill_state_v2(program, execution)), program, execution


def payload(state):
    return build_jev_decision_context_v1(state, translation_step_m=0.01, grasp_xy_tolerance_m=0.008).to_dict()


class SharedSubtaskContextTest(unittest.TestCase):
    def setUp(self):
        self.config = load_runtime_verifier_config_v1(ROOT / "configs/runtime_verifier_v1.yaml")
        self.verifier = RuntimeVerifierV1(self.config)

    def assessed(self, state, program, execution, previous=None):
        result = self.verifier.verify_active_phase(program, execution, state)
        feedback = self.verifier.decision_feedback(program.steps[0], state, result, previous)
        return replace(state, phase_feedback=feedback)

    def test_all_twenty_phases_have_bound_conditions_and_runtime_feedback(self):
        count = 0
        for kind, template in CAPABILITY_TEMPLATES_V1.items():
            for phase in template.phases:
                with self.subTest(kind=kind, phase=phase.phase_id):
                    state, program, execution = phase_state(kind, phase.phase_id)
                    current = self.assessed(state, program, execution)
                    result = payload(current)
                    self.assertEqual(result["phase"]["completion_contract"], phase.completion_contract)
                    self.assertEqual(result["phase"]["verification"]["based_on_state_id"], state.state_id)
                    self.assertEqual(result["phase"]["configured_parameters"], dict(current.phase_feedback.parameters))
                    self.assertTrue(result["phase"]["definition"]["objective"])
                    self.assertTrue(result["phase"]["conditions"])
                    self.assertEqual(set(result["phase"]["measurement_definitions"]),
                        {item["name"] for item in result["phase"]["measurements"]})
                    json.dumps(result)
                    count += 1
        self.assertEqual(count, 20)

    def test_definition_is_reused_but_previous_payload_is_not_mutated(self):
        state, _, _ = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        goal = ("in_region", "left_region", True)
        first = bound_subtask_definition_v1("shared_subject", "surface_push", "track_001", "table_surface", "left_region", goal)
        before = payload(state)
        moved = replace(state, state_id=state.state_id + 1, robot=replace(state.robot,
            tcp_position_robot_base_m=(0.35, 0.05, 0.07)))
        after = payload(moved)
        second = bound_subtask_definition_v1("shared_subject", "surface_push", "track_001", "table_surface", "left_region", goal)
        self.assertIs(first, second)
        self.assertEqual(before["subtask_definition"], after["subtask_definition"])
        self.assertNotEqual(before["interaction"]["subject_from_tcp_robot_base_m"], after["interaction"]["subject_from_tcp_robot_base_m"])
        self.assertEqual(before["robot"]["tcp_position_robot_base_m"][1], 0.0)

    def test_push_goal_and_phase_are_not_collapsed_into_center_error(self):
        state, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        before = payload(self.assessed(state, program, execution))
        outside_goal = replace(state.skill.goal, predicate=VisualPredicate.CLEAR_OF_REGION)
        outside = payload(replace(state, skill=replace(state.skill, goal=outside_goal)))
        retract = payload(replace(state, skill=replace(state.skill, phase="retract_effector")))
        self.assertNotEqual(before["subtask_definition"]["final_goal"], outside["subtask_definition"]["final_goal"])
        self.assertNotEqual(before["phase"]["completion_contract"], retract["phase"]["completion_contract"])
        for item in (before, outside, retract):
            self.assertIsNone(item["interaction"]["goal_delta_robot_base_m"])
        self.assertIn("tcp_outside_subject_footprint_clearance_m", {item["name"] for item in before["phase"]["measurements"]})

    def test_push_height_definition_matches_signed_correction_and_not_an_approach_point(self):
        state, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        state = replace(state, robot=replace(state.robot, tcp_position_robot_base_m=(0.35, 0.0, 0.07)))
        result = payload(self.assessed(state, program, execution))
        values = {item["name"]: item["value"] for item in result["phase"]["measurements"]}
        self.assertAlmostEqual(values["tcp_to_push_height_z_m"], -0.05)
        meaning = result["phase"]["measurement_definitions"]["tcp_to_push_height_z_m"]
        self.assertIn("- measured TCP Z", meaning)
        self.assertIn("Negative means TCP is above", meaning)
        self.assertIsNone(result["interaction"]["goal_delta_robot_base_m"])

    def test_point_alignment_has_signed_xy_correction_and_retains_normal(self):
        state, _, _ = phase_state(CapabilityKind.PRESS, "align_press_point")
        point = replace(state.visual_scene.regions[1], region_type=RegionType.POINT,
            owner_entity_ref="track_001", geometry=replace(state.visual_scene.regions[1].geometry,
            boundary_xy_robot_base_m=None, surface_height_robot_base_m=None,
            point_robot_base_m=measured((0.42, -0.12, 0.055))))
        state = replace(state, visual_scene=replace(state.visual_scene,
            regions=(state.visual_scene.regions[0], point)))
        result = payload(state)
        self.assertAlmostEqual(result["interaction"]["goal_delta_robot_base_m"][0], 0.07)
        self.assertAlmostEqual(result["interaction"]["goal_delta_robot_base_m"][1], -0.12)
        self.assertEqual(result["interaction"]["goal_delta_robot_base_m"][2], 0.0)
        self.assertEqual(result["target"]["surface_normal_robot_base"]["value"], [0.0, 0.0, 1.0])

    def test_unknown_point_is_not_zero_error(self):
        state, _, _ = phase_state(CapabilityKind.PRESS, "align_press_point")
        result = payload(state)  # target is a region, not an observable POINT.
        self.assertIsNone(result["interaction"]["goal_delta_robot_base_m"])
        self.assertIsNone(result["interaction"]["distance_m"])
        self.assertFalse(result["interaction"]["motion_error_observable"])
        self.assertIsNone(result["physical"]["contact"]["left_finger"])

    def test_pressing_does_not_mislabel_xy_constraint_as_pressing_error(self):
        state, _, _ = phase_state(CapabilityKind.PRESS, "press_button")
        result = payload(state)
        self.assertIsNone(result["interaction"]["goal_delta_robot_base_m"])
        self.assertEqual(result["interaction"]["active_axes"], ["x", "y", "z"])

    def test_missing_reference_footprint_is_unknown_in_transport_feedback(self):
        state, program, execution = phase_state(CapabilityKind.TRANSPORT, "align_target_xy", VisualPredicate.ON_SURFACE)
        # A region cannot supply the entity footprint required by ON_SURFACE.
        result = payload(self.assessed(state, program, execution))
        measurements = {item["name"]: item["value"] for item in result["phase"]["measurements"]}
        self.assertIsNone(measurements["placement_horizontal_relation_satisfied"])

    def test_unknown_subject_position_does_not_become_a_zero_offset(self):
        from jev4mujoco.runtime.state_builder import LiveStateSnapshotBuilderV2
        state, _, _ = phase_state(CapabilityKind.PICK, "align_subject_xy")
        entity = state.visual_scene.entities[0]
        unknown = EvidenceValueV1(None, EvidenceStatus.UNOBSERVABLE, 0.0)
        entity = replace(entity, geometry=replace(entity.geometry,
            centroid_robot_base_m=unknown, top_center_robot_base_m=unknown))
        vector, _, relation = LiveStateSnapshotBuilderV2._interaction(state.skill, entity, None,
            state.robot.tcp_position_robot_base_m)
        self.assertIsNone(vector)
        self.assertEqual(relation, "unobservable")
        result = payload(replace(state, visual_scene=replace(state.visual_scene, entities=(entity,))))
        self.assertIsNone(result["interaction"]["goal_delta_robot_base_m"])
        self.assertIsNone(result["interaction"]["subject_from_tcp_robot_base_m"])

    def test_grasp_pose_projection_and_verifier_share_centroid_band_geometry(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "descend_to_grasp")
        entity = state.visual_scene.entities[0]
        entity = replace(entity, geometry=replace(entity.geometry,
            top_center_robot_base_m=measured((0.34, 0.04, 0.06))))
        state = replace(state, visual_scene=replace(state.visual_scene, entities=(entity,)),
            robot=replace(state.robot, tcp_position_robot_base_m=(0.30, -0.10, 0.03)))
        state = self.assessed(state, program, execution)
        self.assertEqual(state.phase_feedback.outcome, "passed")
        self.assertEqual(payload(state)["interaction"]["goal_delta_robot_base_m"], [0.0, 0.0, 0.0])

    def test_inside_height_tolerance_is_named_and_shared_with_visual_relation(self):
        from jev4mujoco.perception.fixed_rgbd import MujocoFixedRgbdVisualSceneBuilderV1
        state, _, _ = phase_state(CapabilityKind.TRANSPORT, "align_target_xy", VisualPredicate.INSIDE)
        entity = state.visual_scene.entities[0]
        volume = replace(state.visual_scene.regions[1], region_type=RegionType.VOLUME_3D,
            owner_entity_ref="track_001", geometry=replace(state.visual_scene.regions[1].geometry,
                boundary_xy_robot_base_m=measured(((0.20, -0.20), (0.40, -0.20), (0.40, 0.0), (0.20, 0.0))),
                minimum_z_robot_base_m=measured(0.005), maximum_z_robot_base_m=measured(0.07)))
        for tolerance, expected in ((0.010, True), (0.0, False)):
            relations = MujocoFixedRgbdVisualSceneBuilderV1._derive_relations((entity,), (volume,),
                relation_config=replace(self.config.visual_relation_config, volume_height_tolerance_m=tolerance))
            relation = next(item for item in relations if item.predicate is VisualPredicate.INSIDE)
            self.assertEqual(relation.value, expected)

    def test_stale_feedback_cannot_be_attached_to_new_state(self):
        state, program, execution = phase_state(CapabilityKind.PRESS, "align_press_point")
        state = self.assessed(state, program, execution)
        with self.assertRaisesRegex(ValueError, "phase feedback"):
            replace(state, state_id=state.state_id + 1)

    def test_previous_measurements_require_matching_phase_and_action(self):
        state, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        before = self.assessed(state, program, execution).phase_feedback
        moved = replace(state, state_id=state.state_id + 1, action_epoch=state.action_epoch + 1,
            recent_action=replace(state.recent_action, start_action_epoch=state.action_epoch,
                end_action_epoch=state.action_epoch + 1),
            robot=replace(state.robot, tcp_position_robot_base_m=(0.35, -0.05, 0.07)))
        current = self.assessed(moved, program, execution, before)
        self.assertTrue(current.phase_feedback.previous_measurements)
        self.assertNotEqual(current.phase_feedback.previous_measurements, current.phase_feedback.measurements)
        wrong_phase = replace(before, phase="establish_contact")
        self.assertFalse(self.assessed(moved, program, execution, wrong_phase).phase_feedback.previous_measurements)

    def test_precontact_accepts_multiple_sides_with_same_geometric_requirements(self):
        state, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        contact = replace(state.physical.effector_contact, mode=EffectorContactMode.CLEAR,
            left_finger_contact=False, right_finger_contact=False)
        state = replace(state, physical=replace(state.physical, effector_contact=contact))
        for tcp in ((0.24, -0.10, 0.02), (0.36, -0.10, 0.02),
                    (0.30, -0.16, 0.02), (0.30, -0.04, 0.02)):
            with self.subTest(tcp=tcp):
                candidate = replace(state, robot=replace(state.robot, tcp_position_robot_base_m=tcp))
                result = self.verifier.verify_active_phase(program, execution, candidate)
                self.assertEqual(result.outcome, VerificationOutcome.PASSED)
                feedback = self.verifier.decision_feedback(program.steps[0], candidate, result)
                self.assertTrue(all(item.satisfied for item in feedback.conditions))
        for tcp in ((0.30, -0.10, 0.02), (0.24, -0.10, 0.20), (0.10, -0.10, 0.02)):
            candidate = replace(state, robot=replace(state.robot, tcp_position_robot_base_m=tcp))
            self.assertEqual(self.verifier.verify_active_phase(program, execution, candidate).outcome, VerificationOutcome.PENDING)

    def test_precontact_uses_observed_polygon_not_axis_aligned_extent(self):
        state, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        entity = state.visual_scene.entities[0]
        entity = replace(entity, geometry=replace(entity.geometry,
            footprint_xy_robot_base_m=measured(((0.30, -0.13), (0.33, -0.10), (0.30, -0.07), (0.27, -0.10)))))
        state = replace(state, visual_scene=replace(state.visual_scene, entities=(entity,)),
            physical=replace(state.physical, effector_contact=replace(state.physical.effector_contact, mode=EffectorContactMode.CLEAR)),
            robot=replace(state.robot, tcp_position_robot_base_m=(0.36, -0.10, 0.02)))
        self.assertEqual(self.verifier.verify_active_phase(program, execution, state).outcome, VerificationOutcome.PASSED)

    def test_relative_goal_keeps_predicate_and_has_no_center_coincidence_command(self):
        state, _, _ = phase_state(CapabilityKind.TRANSPORT, "align_target_xy", VisualPredicate.LEFT_OF)
        reference = replace(state.visual_scene.entities[0], track_id="reference_block")
        state = replace(state,
            visual_scene=replace(state.visual_scene, entities=(*state.visual_scene.entities, reference)),
            skill=replace(state.skill, target_ref="reference_block",
                goal=SkillGoalV1(VisualPredicate.LEFT_OF, "reference_block", True)))
        result = payload(state)
        self.assertEqual(result["subtask_definition"]["final_goal"]["predicate"], "left_of")
        self.assertIsNone(result["interaction"]["goal_delta_robot_base_m"])
        self.assertEqual(result["interaction"]["active_axes"], ["x", "y"])
        self.assertTrue(result["dependencies"]["regions"])

    def test_container_owner_and_source_support_match_clearance_assessment(self):
        state, program, execution = phase_state(CapabilityKind.TRANSPORT, "lift_to_clearance", VisualPredicate.INSIDE)
        owner = replace(state.visual_scene.entities[0], track_id="container",
            geometry=replace(state.visual_scene.entities[0].geometry,
                top_center_robot_base_m=measured((0.275, 0.125, 0.08))))
        volume = replace(state.visual_scene.regions[1], region_type=RegionType.VOLUME_3D,
            owner_entity_ref="container", geometry=replace(state.visual_scene.regions[1].geometry,
                minimum_z_robot_base_m=measured(0.01), maximum_z_robot_base_m=measured(0.07)))
        state = replace(state, visual_scene=replace(state.visual_scene,
            entities=(*state.visual_scene.entities, owner), regions=(state.visual_scene.regions[0], volume)))
        result = payload(self.assessed(state, program, execution))
        self.assertIn("container", {item["ref"] for item in result["dependencies"]["entities"]})
        measurements = {item["name"]: item["value"] for item in result["phase"]["measurements"]}
        self.assertAlmostEqual(measurements["subject_bottom_clearance_deficit_m"], 0.095)
        self.assertEqual(result["interaction"]["goal_delta_robot_base_m"], [0.0, 0.0, 0.095])

    def test_live_loop_passes_actual_pending_verification_to_jev(self):
        from jev4mujoco.policies.jev import CapabilityJevPolicyV1
        from jev4mujoco.runtime.loop import BlockPickPlaceRuntimeV1

        class CaptureClient:
            def __init__(self):
                self.requests = []

            def query(self, request):
                self.requests.append(request)
                return response("blocked", tuple(request["questions"]["decision"]["criteria"]))

        client = CaptureClient()
        with BlockPickPlaceRuntimeV1(ROOT) as runtime:
            runtime.install_policy(CapabilityJevPolicyV1(client))
            outcome = runtime.run(maximum_decisions=1, maximum_retries=0)
        self.assertFalse(outcome.overall_success)  # wiring test, not task execution evidence.
        self.assertEqual(len(client.requests), 1)
        state = client.requests[0]["state"]
        feedback = state["phase"]["verification"]
        self.assertEqual(state["phase"]["phase_id"], "align_subject_xy")
        self.assertEqual(feedback["outcome"], "pending")
        self.assertEqual(feedback["based_on_state_id"], state["state_id"])
        self.assertIn("XY error", feedback["reason"])
        self.assertEqual(state["physical"]["evidence_sources"]["contact"], "mujoco_contact_oracle")
        self.assertEqual(state["phase"]["configured_parameters"]["grasp_xy_tolerance_m"], runtime.grasp_xy_tolerance_m)

    def test_point_alignment_uses_same_rejection_and_measured_progress_as_pick(self):
        from jev4mujoco.policies.jev import CapabilityJevPolicyV1
        from jev4mujoco.experiments.scenarios import LargeButtonPressRuntimeV1
        from tests.test_capability_jev_policy_v1 import FakeJevClient
        prepare_choices = ("gripper_close", "hold", "blocked")
        align_choices = ("forward", "backward", "left", "right", "hold", "blocked")
        client = FakeJevClient([
            response("gripper_close", prepare_choices),
            response("left", align_choices),
            response("right", align_choices),
        ])
        with LargeButtonPressRuntimeV1(ROOT) as runtime:
            runtime.install_policy(CapabilityJevPolicyV1(client))
            outcome = runtime.run(maximum_decisions=3, maximum_retries=0)
        reviews = [event.payload for event in outcome.events if event.kind == "decision_review"]
        self.assertEqual([item.status for item in reviews], ["rejected", "accepted", "observed"])
        actions = [event.payload.action.value for event in outcome.events if event.kind == "action_result"]
        self.assertEqual(actions, ["gripper_close", "right"])
        self.assertGreater(reviews[-1].progress_m, 0.0)
        feedback = client.requests[-1]["state"]["decision_feedback"]
        self.assertEqual(feedback["status"], "rejected")
        self.assertEqual(feedback["action"], "left")
        self.assertEqual(client.requests[-1]["state"]["action_epoch"], client.requests[-2]["state"]["action_epoch"])

    def test_unavailable_support_and_contact_remain_unknown(self):
        state, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        state = replace(state, physical=replace(state.physical,
            support_state=replace(state.physical.support_state, supported=None, support_region_ref=None),
            effector_contact=replace(state.physical.effector_contact, left_finger_contact=None,
                right_finger_contact=None, support_contact=None)))
        result = payload(self.assessed(state, program, execution))
        self.assertIsNone(result["physical"]["support"]["supported"])
        conditions = [item for item in result["phase"]["conditions"] if item["measurement"] == "subject_supported"]
        self.assertIsNone(conditions[0]["observed"])
        self.assertIsNone(conditions[0]["satisfied"])


if __name__ == "__main__":
    unittest.main()
