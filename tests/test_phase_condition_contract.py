from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace

from jev4mujoco.contracts.state_snapshot import EffectorContactMode, GraspState
from jev4mujoco.contracts.visual_scene import (
    EvidenceStatus, RegionType, VisualPredicate, VisualRelationV1,
)
from jev4mujoco.planning.capabilities import CapabilityKind, VerificationOutcome
from jev4mujoco.policies.jev_context import build_local_intent_motor_context_v1
from jev4mujoco.runtime.local_intent import LocalIntentRuntimeV1, load_local_intent_config_v1
from jev4mujoco.runtime.verifier import RuntimeVerifierV1
from jev4mujoco.runtime.verifier_config import load_runtime_verifier_config_v1
from tests.test_shared_subtask_context import ROOT, payload, phase_state
from tests.test_visual_scene_v1 import measured


class PhaseConditionContractTest(unittest.TestCase):
    def setUp(self):
        self.config = load_runtime_verifier_config_v1(ROOT / "configs/runtime_verifier_v1.yaml")
        self.verifier = RuntimeVerifierV1(self.config)

    def assess(self, state, program, execution):
        result = self.verifier.verify_active_phase(program, execution, state)
        feedback = self.verifier.decision_feedback(program.steps[0], state, result)
        return result, feedback, replace(state, phase_feedback=feedback)

    @staticmethod
    def constraints(feedback):
        return {item.constraint_id: item for item in feedback.maintain_constraints}

    @staticmethod
    def grasp(state, status):
        active = status in (GraspState.CANDIDATE_HELD, GraspState.HELD)
        return replace(state, physical=replace(state.physical, grasp_state=status,
            held_object_ref=state.skill.subject_ref if status is GraspState.HELD else None,
            effector_contact=replace(state.physical.effector_contact,
                mode=EffectorContactMode.GRASP_CONTACT if active else EffectorContactMode.UNKNOWN,
                subject_ref=state.skill.subject_ref if active else None,
                left_finger_contact=active, right_finger_contact=active)))

    def press_state(self):
        state, program, execution = phase_state(CapabilityKind.PRESS, "press_button")
        point = replace(state.visual_scene.regions[1], region_type=RegionType.POINT,
            owner_entity_ref=state.skill.subject_ref, geometry=replace(state.visual_scene.regions[1].geometry,
                boundary_xy_robot_base_m=None, surface_height_robot_base_m=None,
                point_robot_base_m=measured((0.42, -0.12, 0.055))))
        return replace(state, visual_scene=replace(state.visual_scene,
            regions=(state.visual_scene.regions[0], point)),
            robot=replace(state.robot, tcp_position_robot_base_m=(0.42, -0.12, 0.02))), program, execution

    def test_candidate_does_not_certify_held_and_unknown_stays_unknown(self):
        for kind, phase, constraint in (
            (CapabilityKind.PICK, "probe_grasp", "candidate_grasp"),
            (CapabilityKind.TRANSPORT, "align_target_xy", "subject_held"),
        ):
            state, program, execution = phase_state(kind, phase)
            for status, expected in ((GraspState.CANDIDATE_HELD,
                "satisfied" if kind is CapabilityKind.PICK else "violated"),
                (GraspState.HELD, "satisfied"), (GraspState.UNKNOWN, "unknown")):
                with self.subTest(kind=kind, grasp=status):
                    _, feedback, _ = self.assess(self.grasp(state, status), program, execution)
                    self.assertEqual(self.constraints(feedback)[constraint].status, expected)

    def test_foreign_candidate_cannot_certify_current_subject(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "probe_grasp")
        state = self.grasp(state, GraspState.CANDIDATE_HELD)
        other = replace(state.visual_scene.entities[0], track_id="track_002")
        state = replace(state, visual_scene=replace(state.visual_scene,
            entities=(*state.visual_scene.entities, other)), physical=replace(state.physical,
            effector_contact=replace(state.physical.effector_contact, subject_ref="track_002")))
        _, feedback, _ = self.assess(state, program, execution)
        candidate = self.constraints(feedback)["candidate_grasp"]
        self.assertEqual(candidate.status, "unknown")
        self.assertIn("not bound", candidate.reason)
        # 原测量/判定未被新诊断规则改写。
        self.assertTrue(next(m.value for m in feedback.measurements if m.name == "grasp_candidate"))

    def test_not_held_does_not_accept_candidate_or_unknown(self):
        state, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "establish_contact")
        for status, expected in ((GraspState.NOT_HELD, "satisfied"),
            (GraspState.CANDIDATE_HELD, "violated"), (GraspState.UNKNOWN, "unknown")):
            with self.subTest(grasp=status):
                _, feedback, _ = self.assess(self.grasp(state, status), program, execution)
                self.assertEqual(self.constraints(feedback)["subject_not_held"].status, expected)

    def test_contact_goal_then_maintenance_then_retraction_have_distinct_scopes(self):
        for phase in ("establish_contact", "push_toward_goal", "retract_effector"):
            state, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, phase)
            state = replace(state, physical=replace(state.physical,
                effector_contact=replace(state.physical.effector_contact, mode=EffectorContactMode.CLEAR,
                    left_finger_contact=False, right_finger_contact=False)))
            _, feedback, _ = self.assess(state, program, execution)
            constraints = self.constraints(feedback)
            if phase == "establish_contact":
                self.assertNotIn("supported_push_contact", constraints)
                self.assertFalse(next(c.satisfied for c in feedback.conditions
                    if c.measurement == "supported_push_contact"))
            elif phase == "push_toward_goal":
                self.assertEqual(constraints["supported_push_contact"].status, "violated")
            else:
                self.assertNotIn("supported_push_contact", constraints)
                self.assertTrue(next(c.satisfied for c in feedback.conditions
                    if c.measurement == "tool_subject_contact_clear"))

    def test_release_command_and_loss_of_holding_do_not_end_relation_scope(self):
        state, program, execution = phase_state(CapabilityKind.PLACE, "lower_to_release")
        relation = VisualRelationV1(VisualPredicate.IN_REGION, state.skill.subject_ref,
            state.skill.goal.reference_ref, False, EvidenceStatus.DERIVED, 1.0)
        state = replace(state, visual_scene=replace(state.visual_scene, relations=(relation,)))
        state = self.grasp(state, GraspState.HELD)
        state = replace(state, robot=replace(state.robot,
            gripper=replace(state.robot.gripper, opening_m=0.02,
                target_opening_m=self.config.gripper_opening_m)))
        for current in (state, self.grasp(state, GraspState.NOT_HELD)):
            result, feedback, _ = self.assess(current, program, execution)
            self.assertIs(result.outcome, VerificationOutcome.PENDING)
            constraint = self.constraints(feedback)["placement_relation_until_release"]
            self.assertEqual(constraint.scope, "until_observed_release")
            self.assertEqual(constraint.status, "violated")
        released = self.grasp(state, GraspState.NOT_HELD)
        released = replace(released, robot=replace(released.robot,
            gripper=replace(released.robot.gripper, opening_m=self.config.gripper_opening_m)))
        result, feedback, _ = self.assess(released, program, execution)
        self.assertIs(result.outcome, VerificationOutcome.PASSED)
        self.assertEqual(self.constraints(feedback)["placement_relation_until_release"].status, "not_applicable")

    def test_unknown_grasp_source_cannot_certify_observed_release(self):
        state, program, execution = phase_state(CapabilityKind.PLACE, "lower_to_release")
        state = replace(state, physical=replace(state.physical,
            evidence=replace(state.physical.evidence, grasp_source="unknown")))
        result, feedback, _ = self.assess(state, program, execution)
        self.assertIs(result.outcome, VerificationOutcome.PASSED)  # 原判定保持不变。
        self.assertIsNone(next(c.satisfied for c in feedback.conditions if c.measurement == "grasp_not_held"))
        self.assertNotEqual(self.constraints(feedback)["placement_relation_until_release"].status, "not_applicable")

    def test_old_robot_measurement_cannot_be_refreshed_by_new_visual_without_action(self):
        state, program, execution = phase_state(CapabilityKind.PLACE, "lower_to_release")
        state = replace(state, recent_action=None,
            sources=replace(state.sources, robot_measurement_timestamp_s=0.0))
        result, feedback, _ = self.assess(state, program, execution)
        self.assertIs(result.outcome, VerificationOutcome.PASSED)
        self.assertFalse(feedback.evidence_basis.robot_evidence_current)
        self.assertIsNone(next(c.satisfied for c in feedback.conditions if c.measurement == "gripper_position_m"))
        self.assertNotEqual(self.constraints(feedback)["placement_relation_until_release"].status, "not_applicable")

    def test_press_alignment_uses_existing_tolerance_and_unknown_owned_point(self):
        state, program, execution = self.press_state()
        for x, expected in ((0.42, "satisfied"),
            (0.42 + 2 * self.config.press_point_xy_tolerance_m, "violated")):
            current = replace(state, robot=replace(state.robot, tcp_position_robot_base_m=(x, -0.12, 0.02)))
            _, feedback, _ = self.assess(current, program, execution)
            self.assertEqual(self.constraints(feedback)["press_point_alignment"].status, expected)
        point = state.visual_scene.regions[1]
        # POINT schema 要求已知位置；保留目标引用但缺少点几何的合法输入。
        point = replace(point, region_type=RegionType.REGION_2D, geometry=replace(point.geometry,
            point_robot_base_m=None, surface_height_robot_base_m=measured(0.055), boundary_xy_robot_base_m=measured(
                ((0.40, -0.14), (0.44, -0.14), (0.44, -0.10), (0.40, -0.10)))))
        state = replace(state, visual_scene=replace(state.visual_scene,
            regions=(state.visual_scene.regions[0], point)))
        _, feedback, _ = self.assess(state, program, execution)
        self.assertEqual(self.constraints(feedback)["press_point_alignment"].status, "unknown")

    def test_press_point_with_wrong_owner_cannot_certify_constraint(self):
        state, program, execution = self.press_state()
        other = replace(state.visual_scene.entities[0], track_id="track_002")
        point = replace(state.visual_scene.regions[1], owner_entity_ref="track_002")
        state = replace(state, visual_scene=replace(state.visual_scene,
            entities=(*state.visual_scene.entities, other), regions=(state.visual_scene.regions[0], point)))
        _, feedback, _ = self.assess(state, program, execution)
        self.assertEqual(self.constraints(feedback)["press_point_alignment"].status, "unknown")

    def test_stale_visual_preserves_raw_measurement_but_cannot_certify_current_conditions(self):
        state, program, execution = self.press_state()
        timestamp = state.timestamp_s + self.config.max_visual_age_s + 0.01
        state = replace(state, timestamp_s=timestamp,
            sources=replace(state.sources, robot_measurement_timestamp_s=timestamp),
            freshness=replace(state.freshness, visual_age_s=timestamp - state.visual_scene.capture_timestamp_s))
        result, feedback, current = self.assess(state, program, execution)
        self.assertIs(result.outcome, VerificationOutcome.PENDING)
        self.assertEqual(next(m.value for m in feedback.measurements if m.name == "tcp_press_point_xy_error_m"), 0.0)
        self.assertIsNone(next(c.satisfied for c in feedback.conditions if c.measurement == "tcp_press_point_xy_error_m"))
        self.assertEqual(self.constraints(feedback)["press_point_alignment"].status, "unknown")
        json.dumps(current.to_dict())

    def test_visual_predating_latest_action_is_not_current_even_within_age_limit(self):
        state, program, execution = self.press_state()
        timestamp = state.timestamp_s + min(0.01, self.config.max_visual_age_s / 2)
        state = replace(state, timestamp_s=timestamp,
            sources=replace(state.sources, robot_measurement_timestamp_s=timestamp),
            freshness=replace(state.freshness, visual_age_s=timestamp - state.visual_scene.capture_timestamp_s),
            recent_action=replace(state.recent_action, end_timestamp_s=timestamp))
        _, feedback, _ = self.assess(state, program, execution)
        self.assertFalse(feedback.evidence_basis.visual_evidence_current)
        self.assertEqual(self.constraints(feedback)["press_point_alignment"].status, "unknown")

    def test_support_needs_current_subject_binding_and_preserves_phase_verdict(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "align_subject_xy")
        other = replace(state.visual_scene.entities[0], track_id="track_002")
        state = replace(state, visual_scene=replace(state.visual_scene,
            entities=(*state.visual_scene.entities, other)),
            robot=replace(state.robot, tcp_position_robot_base_m=(0.30, -0.10, 0.07)),
            physical=replace(state.physical,
                support_state=replace(state.physical.support_state, subject_ref="track_002")))
        result, feedback, _ = self.assess(state, program, execution)
        self.assertIs(result.outcome, VerificationOutcome.PASSED)
        self.assertEqual(self.constraints(feedback)["subject_supported"].status, "unknown")

    def test_supported_phase_does_not_gain_a_collision_safety_certificate(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "align_subject_xy")
        state = replace(state, robot=replace(state.robot, tcp_position_robot_base_m=(0.30, -0.10, 0.07)))
        result, feedback, _ = self.assess(state, program, execution)
        self.assertIs(result.outcome, VerificationOutcome.PASSED)
        constraint = self.constraints(feedback)["avoid_other_objects"]
        self.assertEqual(constraint.status, "unsupported")
        self.assertIsNone(constraint.condition)
        self.assertFalse(constraint.evidence)

    def test_satisfied_xy_goal_does_not_erase_violated_maintenance(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "align_subject_xy")
        state = replace(state, robot=replace(state.robot, tcp_position_robot_base_m=(0.30, -0.10, 0.07)),
            physical=replace(state.physical,
                support_state=replace(state.physical.support_state, supported=False, support_region_ref=None)))
        result, feedback, _ = self.assess(state, program, execution)
        self.assertIs(result.outcome, VerificationOutcome.PASSED)
        self.assertEqual(feedback.outcome, "passed")
        self.assertEqual(self.constraints(feedback)["subject_supported"].status, "violated")

    def test_unknown_support_and_unavailable_robot_feedback_remain_unknown(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "align_subject_xy")
        for physical in (
            replace(state.physical, support_state=replace(state.physical.support_state,
                supported=None, support_region_ref=None)),
            replace(state.physical, evidence=replace(state.physical.evidence, robot_feedback_available=False)),
        ):
            _, feedback, _ = self.assess(replace(state, physical=physical), program, execution)
            self.assertEqual(self.constraints(feedback)["subject_supported"].status, "unknown")

    def test_result_cannot_be_rebound_to_new_state_epoch_observation_or_phase(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "align_subject_xy")
        result = self.verifier.verify_active_phase(program, execution, state)
        newer_scene = replace(state.visual_scene, observation_id=state.visual_scene.observation_id + 1)
        variants = (
            replace(state, state_id=state.state_id + 1),
            replace(state, action_epoch=state.action_epoch + 1, recent_action=None),
            replace(state, visual_scene=newer_scene,
                sources=replace(state.sources, visual_observation_id=newer_scene.observation_id)),
            replace(state, skill=replace(state.skill, phase="descend_to_grasp")),
        )
        for current in variants:
            with self.subTest(state=current.state_id, epoch=current.action_epoch, phase=current.skill.phase):
                with self.assertRaisesRegex(ValueError, "current bound phase"):
                    self.verifier.decision_feedback(program.steps[0], current, result)

    def test_feedback_cannot_refresh_evidence_timestamps_while_reusing_state_id(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "align_subject_xy")
        _, feedback, current = self.assess(state, program, execution)
        with self.assertRaisesRegex(ValueError, "evidence timestamps"):
            replace(current, phase_feedback=replace(feedback, evidence_basis=replace(feedback.evidence_basis,
                robot_measurement_timestamp_s=0.0)))

    def test_result_scope_cannot_be_rebound_even_when_state_epoch_and_observation_match(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "align_subject_xy")
        state = replace(state, robot=replace(state.robot, tcp_position_robot_base_m=(0.30, -0.10, 0.07)))
        result = self.verifier.verify_active_phase(program, execution, state)
        self.assertIs(result.outcome, VerificationOutcome.PASSED)
        for field, instance_field, value in (
            ("skill_instance_id", "instance_id", "another_capability"),
            ("source_ref", "source_ref", "left_region"),
        ):
            with self.subTest(field=field):
                current = replace(state, skill=replace(state.skill, **{field: value}))
                instance = replace(program.steps[0], **{instance_field: value})
                with self.assertRaisesRegex(ValueError, "current bound phase"):
                    self.verifier.decision_feedback(instance, current, result)
        _, feedback, current = self.assess(state, program, execution)
        with self.assertRaisesRegex(ValueError, "feedback scope"):
            replace(current, skill=replace(current.skill, source_ref="left_region"))

    def test_point_motor_retains_all_conditions_and_independent_phase_outcome(self):
        state, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        _, feedback, current = self.assess(state, program, execution)
        runtime = LocalIntentRuntimeV1(load_local_intent_config_v1(ROOT / "configs/local_intent_v1.yaml"))
        current = replace(current, local_intent=runtime.context(current))
        candidate = next(c for c in current.local_intent.candidates if c.choice_id == "adjust_push_height")
        full = payload(current)
        before = copy.deepcopy(full)
        motor = build_local_intent_motor_context_v1(current, candidate, full)
        self.assertEqual(motor["phase"]["conditions"], full["phase"]["conditions"])
        self.assertEqual(motor["phase"]["maintain_constraints"], full["phase"]["maintain_constraints"])
        self.assertEqual(motor["phase"]["evidence_basis"], full["phase"]["evidence_basis"])
        self.assertEqual(motor["phase"]["runtime_outcome"], feedback.outcome)
        self.assertEqual(motor["phase"]["condition_role"], "completion_goals_not_action_preconditions")
        evaluations = {e["action"]: e for e in motor["action_evaluations"]}
        self.assertGreater(evaluations["down"]["projected_progress_m"], 0)
        self.assertEqual(feedback.outcome, "pending")  # 预测进展不是完成。
        motor["phase"]["conditions"][0]["observed"] = 999.0
        self.assertEqual(full, before)

    def test_projection_does_not_increment_stable_history_or_skip_observation_interval(self):
        state, program, execution = phase_state(CapabilityKind.PLACE, "verify_placement")
        relation = VisualRelationV1(VisualPredicate.IN_REGION, state.skill.subject_ref,
            state.skill.goal.reference_ref, True, EvidenceStatus.DERIVED, 1.0)
        state = replace(state, visual_scene=replace(state.visual_scene, relations=(relation,)))
        result = self.verifier.verify_active_phase(program, execution, state)
        self.assertEqual(result.consecutive_pass_count, 1)
        history = copy.deepcopy(self.verifier._placement_history)
        for _ in range(3):
            feedback = self.verifier.decision_feedback(program.steps[0], state, result)
            self.assertEqual(feedback.consecutive_pass_count, 1)
            self.assertEqual(self.verifier._placement_history, history)
        interval = self.config.placement_minimum_observation_interval_s
        for offset, expected_count in ((interval / 2, 1), (interval + 0.001, 2)):
            timestamp = state.timestamp_s + offset
            scene = replace(state.visual_scene, observation_id=state.visual_scene.observation_id + 1,
                capture_timestamp_s=timestamp, publish_timestamp_s=timestamp)
            newer = replace(state, state_id=state.state_id + 1, timestamp_s=timestamp, visual_scene=scene,
                sources=replace(state.sources, visual_observation_id=scene.observation_id,
                    visual_capture_timestamp_s=timestamp, robot_measurement_timestamp_s=timestamp))
            result, feedback, _ = self.assess(newer, program, execution)
            self.assertEqual(result.consecutive_pass_count, expected_count)
            self.assertEqual(feedback.consecutive_pass_count, expected_count)
            if offset < interval:
                self.assertIs(result.outcome, VerificationOutcome.PENDING)
                self.assertIn("interval", result.reason)


if __name__ == "__main__":
    unittest.main()
