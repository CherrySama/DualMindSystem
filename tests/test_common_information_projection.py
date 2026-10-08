from dataclasses import replace
import unittest
from unittest.mock import patch

from jev4mujoco.contracts.local_intent import LocalIntentChoiceV1
from jev4mujoco.contracts.state_snapshot import GraspState, SkillGoalV1
from jev4mujoco.contracts.visual_scene import VisualPredicate
from jev4mujoco.planning.capabilities import CapabilityKind
from jev4mujoco.policies.jev import CapabilityJevPolicyV1, _phase_interaction_rules
from jev4mujoco.policies.jev_context import build_local_intent_motor_context_v1
from jev4mujoco.runtime.local_intent import LocalIntentRuntimeV1, load_local_intent_config_v1
from tests.test_capability_jev_policy_v1 import FakeJevClient, response
from tests import test_shared_subtask_context as fixtures
from tests.test_shared_subtask_context import ROOT, phase_state, payload


class CommonInformationProjectionTest(unittest.TestCase):
    def setUp(self):
        self.assessor = fixtures.SharedSubtaskContextTest()
        self.assessor.setUp()
        self.config = load_local_intent_config_v1(ROOT / "configs/local_intent_v1.yaml")

    def motor_request(self, kind, phase, intent):
        state, program, execution = phase_state(kind, phase)
        if kind is CapabilityKind.PLACE:
            # 仍持有主体，释放阶段才需要接受动作，不能用已通过的夹具替代。
            state = replace(state, physical=replace(state.physical,
                grasp_state=GraspState.HELD, held_object_ref=state.skill.subject_ref))
        state = self.assessor.assessed(state, program, execution)
        runtime = LocalIntentRuntimeV1(self.config)
        state = replace(state, local_intent=runtime.context(state))
        candidate = next(item for item in state.local_intent.candidates if item.choice_id == intent)
        runtime.accept(LocalIntentChoiceV1(intent, state.state_id, state.action_epoch,
            state.skill.skill_instance_id, state.skill.phase), state)
        state = replace(state, local_intent=runtime.context(state))
        choice = "hold" if "hold" in candidate.allowed_actions else candidate.allowed_actions[0]
        client = FakeJevClient(response(choice, (*candidate.allowed_actions, "replan_local_intent", "blocked")))
        CapabilityJevPolicyV1(client, jev_stages=2).decide(state)
        self.assertEqual(len(client.requests), 1)
        single = FakeJevClient(response(choice, (*state.skill.allowed_actions, "blocked")))
        CapabilityJevPolicyV1(single).decide(state)
        return state, client.requests[0], single.requests[0]

    def test_probe_interaction_survives_motor_projection(self):
        state, motor, single = self.motor_request(CapabilityKind.PICK, "probe_grasp", "probe_grasp")
        mechanism = _phase_interaction_rules(state)
        self.assertIn(mechanism, single["questions"]["decision"]["instructions"]["rules"])
        self.assertIn(mechanism, motor["questions"]["decision"]["instructions"]["rules"])
        self.assertIn("object follows the TCP", mechanism)
        self.assertIn("Runtime alone confirms", mechanism)

    def test_release_explanation_does_not_expand_selected_intent_actions(self):
        for intent, expected in (("lower_to_release", {"up", "down", "hold"}),
                                 ("release_subject", {"gripper_open", "hold"})):
            with self.subTest(intent=intent):
                state, motor, single = self.motor_request(CapabilityKind.PLACE, "lower_to_release", intent)
                mechanism = _phase_interaction_rules(state)
                rules = motor["questions"]["decision"]["instructions"]["rules"]
                self.assertIn(mechanism, rules)
                self.assertIn(mechanism, single["questions"]["decision"]["instructions"]["rules"])
                self.assertIn("signed drop distance", mechanism)
                self.assertIn("not pre-vetoed", mechanism)
                self.assertEqual(set(motor["state"]["handle"]["available_actions"]), expected)
                self.assertEqual(set(motor["questions"]["decision"]["criteria"]),
                    expected | {"replan_local_intent", "blocked"})
                for full_only in ("motion_error_role", "goal_delta", "choose a feasible contact side"):
                    self.assertNotIn(full_only, rules)
                self.assertIn("do not silently reselect", rules)

    def test_mechanism_explanations_are_scoped_to_their_capability_phase(self):
        for kind, phase in ((CapabilityKind.TRANSPORT, "align_target_xy"),
                            (CapabilityKind.SURFACE_PUSH, "push_toward_goal"),
                            (CapabilityKind.PRESS, "press_button"),
                            (CapabilityKind.PICK, "close_gripper")):
            with self.subTest(kind=kind):
                state, _, _ = phase_state(kind, phase)
                self.assertEqual(_phase_interaction_rules(state), "")

    def advance(self, state):
        timestamp = state.timestamp_s + 0.1
        return replace(state, phase_feedback=None, local_intent=None,
            state_id=state.state_id + 1, action_epoch=state.action_epoch + 1, timestamp_s=timestamp,
            sources=replace(state.sources, visual_capture_timestamp_s=timestamp,
                robot_measurement_timestamp_s=timestamp,
                visual_observation_id=state.visual_scene.observation_id + 1),
            visual_scene=replace(state.visual_scene, observation_id=state.visual_scene.observation_id + 1,
                capture_timestamp_s=timestamp, publish_timestamp_s=timestamp),
            freshness=replace(state.freshness, visual_is_fresh=True, visual_age_s=0.0),
            recent_action=replace(state.recent_action, start_action_epoch=state.action_epoch,
                end_action_epoch=state.action_epoch + 1, end_timestamp_s=timestamp - 0.05))

    def history(self, kind=CapabilityKind.PLACE, phase="lower_to_release", before=None, modify=None):
        state, program, execution = phase_state(kind, phase)
        if before is not None:
            state = before(state)
        previous = self.assessor.assessed(state, program, execution).phase_feedback
        current = self.advance(state)
        if modify is not None:
            current = modify(current)
        current = self.assessor.assessed(current, program, execution, previous)
        return previous, current, program, execution

    def test_prior_frame_keeps_original_identity_time_and_measurements_once(self):
        previous, state, _, _ = self.history()
        feedback = state.phase_feedback
        self.assertEqual(feedback.previous_observation_basis, previous.observation_basis)
        self.assertEqual(feedback.previous_measurements, previous.measurements)
        self.assertNotEqual(feedback.observation_basis.timestamp_s,
            feedback.previous_observation_basis.timestamp_s)
        full = payload(state)
        self.assertEqual(full["phase"]["previous_observation_basis"]["observation_id"], previous.based_on_observation_id)
        self.assertEqual(full["phase"]["previous_observation_basis"]["timestamp_s"], previous.observation_basis.timestamp_s)
        self.assertNotIn("previous_measurements", full["phase"]["previous_observation_basis"])
        self.assertNotIn("previous_observation_basis", full["phase"]["previous_observation_basis"])

    def test_all_five_capabilities_pass_applicable_history_to_motor(self):
        phases = ((CapabilityKind.PICK, "align_subject_xy"),
            (CapabilityKind.TRANSPORT, "align_target_xy"), (CapabilityKind.PLACE, "lower_to_release"),
            (CapabilityKind.SURFACE_PUSH, "push_toward_goal"), (CapabilityKind.PRESS, "press_button"))
        for kind, phase in phases:
            with self.subTest(kind=kind):
                previous, state, _, _ = self.history(kind, phase)
                runtime = LocalIntentRuntimeV1(self.config)
                state = replace(state, local_intent=runtime.context(state))
                full = payload(state)
                candidate = state.local_intent.candidates[0]
                motor = build_local_intent_motor_context_v1(state, candidate, full)
                history = motor["phase"]["previous_measurements"]
                self.assertTrue(history)
                self.assertEqual(motor["phase"]["previous_observation_basis"]["state_id"], previous.based_on_state_id)
                originals = {m.name: m for m in previous.measurements}
                for item in history:
                    original = originals[item["name"]]
                    self.assertEqual((item["value"], item["unit"], item["source"], item["confidence"]),
                        (original.value, original.unit, original.source, original.confidence))
                self.assertEqual(set(motor["phase"]["previous_observation_basis"]["measurement_evidence_current"]),
                    {item["name"] for item in history})
                self.assertEqual(set(motor["phase"]["observation_basis"]["measurement_evidence_current"]),
                    {item["name"] for item in motor["phase"]["measurements"]})
                self.assertEqual(motor["phase"]["conditions"], full["phase"]["conditions"])
                self.assertEqual(motor["phase"]["maintain_constraints"], full["phase"]["maintain_constraints"])

    def test_old_visual_invalidity_is_not_refreshed_by_new_fresh_frame(self):
        def stale(state):
            capture = state.timestamp_s - 0.1  # 尚未晚于旧最近动作的结束时间。
            return replace(state, sources=replace(state.sources, visual_capture_timestamp_s=capture),
                visual_scene=replace(state.visual_scene, capture_timestamp_s=capture, publish_timestamp_s=state.timestamp_s),
                freshness=replace(state.freshness, visual_is_fresh=True, visual_age_s=0.1))
        previous, state, _, _ = self.history(CapabilityKind.PICK, "align_subject_xy", before=stale)
        name = "tcp_subject_xy_error_m"
        old = state.phase_feedback.previous_observation_basis
        self.assertEqual(old, previous.observation_basis)
        self.assertFalse(dict(old.measurement_evidence_current)[name])
        self.assertTrue(dict(state.phase_feedback.observation_basis.measurement_evidence_current)[name])
        self.assertLess(old.evidence_basis.visual_capture_timestamp_s, old.preceding_action_end_timestamp_s)

    def test_old_robot_invalidity_and_unknown_source_remain_invalid(self):
        def old_robot(state):
            return replace(state, sources=replace(state.sources,
                robot_measurement_timestamp_s=state.timestamp_s - 0.1))
        previous, state, _, _ = self.history(before=old_robot)
        self.assertFalse(dict(state.phase_feedback.previous_observation_basis.measurement_evidence_current)["gripper_position_m"])
        self.assertTrue(dict(state.phase_feedback.observation_basis.measurement_evidence_current)["gripper_position_m"])
        original = self.assessor.verifier._phase_measurements
        def unknown_source(*args):
            return tuple(replace(m, source="unknown") if m.name == "subject_bottom_to_target_support_clearance_m"
                         else m for m in original(*args))
        with patch.object(self.assessor.verifier, "_phase_measurements", side_effect=unknown_source):
            previous, state, _, _ = self.history()
        self.assertFalse(dict(previous.observation_basis.measurement_evidence_current)["subject_bottom_to_target_support_clearance_m"])
        self.assertFalse(dict(state.phase_feedback.previous_observation_basis.measurement_evidence_current)["subject_bottom_to_target_support_clearance_m"])

    def test_unknown_prior_value_is_preserved_without_valid_comparison(self):
        original = self.assessor.verifier._phase_measurements
        def unknown_value(*args):
            return tuple(replace(m, value=None) if m.name == "subject_bottom_to_target_support_clearance_m"
                         else m for m in original(*args))
        with patch.object(self.assessor.verifier, "_phase_measurements", side_effect=unknown_value):
            previous, state, _, _ = self.history()
        old = next(m for m in state.phase_feedback.previous_measurements
            if m.name == "subject_bottom_to_target_support_clearance_m")
        self.assertIsNone(old.value)
        self.assertFalse(dict(previous.observation_basis.measurement_evidence_current)[old.name])

    def test_same_phase_permission_changes_keep_history_but_scope_changes_do_not(self):
        _, state, _, _ = self.history(modify=lambda s: replace(s,
            skill=replace(s.skill, allowed_actions=("hold",))))
        self.assertTrue(state.phase_feedback.previous_measurements)
        before, state, program, execution = self.history()
        for name, value in (("skill_kind", "transport"), ("phase", "align_target_xy"),
            ("subject_ref", "other_subject"), ("source_ref", "other_support"),
            ("target_ref", "other_target"),
            ("goal", SkillGoalV1(VisualPredicate.CLEAR_OF_REGION, "left_region", True))):
            with self.subTest(name=name):
                changed = replace(before, observation_basis=None,
                    based_on_skill=replace(before.based_on_skill, **{name: value}))
                current = self.assessor.assessed(replace(state, phase_feedback=None), program, execution, changed)
                self.assertFalse(current.phase_feedback.previous_measurements)

    def test_sensor_scene_calibration_and_action_mismatch_drop_history(self):
        def scene_changed(s):
            return replace(s, visual_scene=replace(s.visual_scene, scene_id="another_scene"),
                sources=replace(s.sources, visual_scene_id="another_scene"))
        changes = (scene_changed,
            lambda s: replace(s, visual_scene=replace(s.visual_scene,
                sensor=replace(s.visual_scene.sensor, sensor_id="another_sensor"))),
            lambda s: replace(s, visual_scene=replace(s.visual_scene,
                sensor=replace(s.visual_scene.sensor, calibration_id="another_calibration"))),
            lambda s: replace(s, recent_action=replace(s.recent_action,
                start_action_epoch=s.recent_action.start_action_epoch - 1)),
            lambda s: replace(s, recent_action=replace(s.recent_action, end_timestamp_s=0.0)))
        for modify in changes:
            previous, state, _, _ = self.history(modify=modify)
            self.assertFalse(state.phase_feedback.previous_measurements)

    def test_legacy_basis_missing_is_unverified_and_duplicate_projection_has_no_history_chain(self):
        before, state, program, execution = self.history()
        legacy = replace(before, observation_basis=None)
        current = self.assessor.assessed(replace(state, phase_feedback=None), program, execution, legacy)
        self.assertEqual(current.phase_feedback.previous_measurements, legacy.measurements)
        self.assertIsNone(current.phase_feedback.previous_observation_basis)
        self.assertEqual(payload(current)["phase"]["previous_observation_basis"], None)
        duplicate = self.assessor.assessed(replace(current, phase_feedback=None), program, execution,
            current.phase_feedback)
        self.assertFalse(duplicate.phase_feedback.previous_measurements)
        self.assertIsNone(duplicate.phase_feedback.previous_observation_basis)

    def test_observation_basis_cannot_be_rebound_to_different_frame_or_measurement_set(self):
        before, state, _, _ = self.history()
        for changed in (replace(state.phase_feedback.observation_basis, observation_id=99),
                        replace(state.phase_feedback.observation_basis, measurement_evidence_current=())):
            with self.assertRaisesRegex(ValueError, "observation basis"):
                replace(state.phase_feedback, observation_basis=changed)
        changed = replace(state.phase_feedback.observation_basis, sensor_id="another_sensor")
        with self.assertRaisesRegex(ValueError, "observation basis"):
            replace(state, phase_feedback=replace(state.phase_feedback, observation_basis=changed))


if __name__ == "__main__":
    unittest.main()
