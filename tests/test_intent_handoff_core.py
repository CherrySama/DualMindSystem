"""验证意图结束与事实交接；不把这些契约反例算作任务成功。"""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from jev4mujoco.contracts.local_intent import LocalIntentChoiceV1
from jev4mujoco.contracts.state_snapshot import EffectorContactMode
from jev4mujoco.contracts.visual_scene import Visibility
from jev4mujoco.planning.capabilities import CapabilityKind, VerificationOutcome
from jev4mujoco.policies.jev_context import build_local_intent_motor_context_v1
from jev4mujoco.runtime.local_intent import LocalIntentRuntimeV1, load_local_intent_config_v1
from jev4mujoco.runtime.capability_runtime import CapabilityRuntimeV1, CapabilityRuntimeEventKind
from jev4mujoco.runtime.loop import BlockPickPlaceRuntimeV1
from tests import test_shared_subtask_context as fixtures
from tests.test_shared_subtask_context import ROOT, phase_state, payload
from tests.test_push_contact_round2 import box_tool
from tests.test_visual_scene_v1 import measured


class IntentHandoffCoreTest(unittest.TestCase):
    def setUp(self):
        self.assessor = fixtures.SharedSubtaskContextTest()
        self.assessor.setUp()
        self.runtime = LocalIntentRuntimeV1(load_local_intent_config_v1(ROOT / "configs/local_intent_v1.yaml"))
        self.state, self.program, self.execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")
        self.state = replace(self.state, robot=replace(self.state.robot, tool_geometry=box_tool()),
            physical=replace(self.state.physical, effector_contact=replace(self.state.physical.effector_contact,
                mode=EffectorContactMode.CLEAR)))
        contact, _, _ = phase_state(CapabilityKind.SURFACE_PUSH, "establish_contact")
        self.next_skill = contact.skill

    def assess(self, state):
        return self.assessor.assessed(replace(state, phase_feedback=None, local_intent=None),
            self.program, self.execution)

    def select(self, name="approach_edge_2"):
        state = self.assess(self.state)
        state = replace(state, local_intent=self.runtime.context(state))
        chosen = self.runtime.accept(LocalIntentChoiceV1(name, state.state_id, state.action_epoch,
            state.skill.skill_instance_id, state.skill.phase), state)
        return state, chosen

    def observed(self, state, tcp=None):
        observation = state.visual_scene.observation_id + 1
        return replace(state, state_id=state.state_id + 1, local_intent=None, phase_feedback=None,
            visual_scene=replace(state.visual_scene, observation_id=observation),
            sources=replace(state.sources, visual_observation_id=observation),
            robot=replace(state.robot, tcp_position_robot_base_m=tcp or state.robot.tcp_position_robot_base_m))

    def enter_contact(self, state):
        state = self.observed(state)
        state = replace(state, skill=self.next_skill)
        _, program, execution = phase_state(CapabilityKind.SURFACE_PUSH, "establish_contact")
        state = self.assessor.assessed(state, program, execution)
        return replace(state, local_intent=self.runtime.context(state))

    def reached(self):
        state, chosen = self.select()
        target = chosen.candidate.target_robot_base_m
        state = self.assess(self.observed(state, (*target[:2], 0.02)))
        self.assertEqual(state.phase_feedback.outcome, "passed")
        self.runtime.close_phase(state, self.next_skill)
        return state

    def test_legal_phase_pass_supersedes_unreached_plan_without_replan_cost(self):
        state, chosen = self.select()
        # -Y 外侧合法，JEV 选择的 +X 点尚未到达；Runtime 的通过应保留。
        state = self.assess(self.observed(state, (0.30, -0.16, 0.02)))
        self.assertEqual(state.phase_feedback.outcome, "passed")
        ended = self.runtime.close_phase(state, self.next_skill)
        self.assertEqual(ended.execution.status, "superseded")
        self.assertGreater(ended.execution.metric_m, chosen.candidate.tolerance_m)
        self.assertEqual((ended.based_on_state_id, ended.based_on_action_epoch, ended.based_on_observation_id),
            (state.state_id, state.action_epoch, state.visual_scene.observation_id))
        self.assertEqual(self.runtime.context(state).replan_count, 0)
        self.assertIsNone(self.runtime.close_phase(state, self.next_skill))
        contact = self.enter_contact(state)
        self.assertIsNone(contact.local_intent.execution)
        self.assertIsNone(contact.local_intent.retained_approach)
        self.assertIsNone(contact.local_intent.handoff)
        self.assertEqual(len(contact.local_intent.candidates), 4)
        self.assertEqual(contact.local_intent.last_termination, ended)
        self.assertEqual(len(payload(contact)["push_contact_geometry"]), 4)

    def test_fresh_reach_hands_off_chosen_side_without_asserting_contact(self):
        contact = self.enter_contact(self.reached())
        handoff = contact.local_intent.handoff
        self.assertEqual(handoff.condition, "precontact_xy_reached")
        self.assertEqual(handoff.based_on_observation_id, contact.visual_scene.observation_id)
        self.assertEqual(handoff.source_phase, "align_precontact")
        self.assertEqual(handoff.destination_phase, "establish_contact")
        self.assertEqual(contact.phase_feedback.outcome, "pending")
        self.assertEqual([c.choice_id for c in contact.local_intent.candidates], ["contact_edge_2"])
        self.assertEqual(contact.local_intent.last_termination.execution.status, "completed")
        full = payload(contact)
        self.assertEqual(len(full["push_contact_geometry"]), 1)
        motor = build_local_intent_motor_context_v1(contact, contact.local_intent.candidates[0], full)
        self.assertEqual(motor["local_intent"]["handoff"], full["local_intent"]["handoff"])
        self.assertEqual(motor["local_intent"]["last_termination"], full["local_intent"]["last_termination"])

    def test_reached_xy_then_height_adjustment_preserves_handoff(self):
        state, chosen = self.select()
        state = self.assess(self.observed(state, chosen.candidate.target_robot_base_m))
        state = replace(state, local_intent=self.runtime.context(state))
        self.assertEqual(state.local_intent.execution.status, "completed")
        self.assertEqual(state.phase_feedback.outcome, "pending")
        self.runtime.accept(LocalIntentChoiceV1("adjust_push_height", state.state_id, state.action_epoch,
            state.skill.skill_instance_id, state.skill.phase), state)
        state = self.assess(self.observed(state, (*chosen.candidate.target_robot_base_m[:2], 0.02)))
        self.runtime.close_phase(state, self.next_skill)
        self.assertIsNotNone(self.enter_contact(state).local_intent.handoff)

    def test_earlier_completed_xy_is_not_proof_after_height_action_drift(self):
        state, chosen = self.select()
        state = self.assess(self.observed(state, chosen.candidate.target_robot_base_m))
        state = replace(state, local_intent=self.runtime.context(state))
        self.assertEqual(state.local_intent.execution.status, "completed")
        self.runtime.accept(LocalIntentChoiceV1("adjust_push_height", state.state_id, state.action_epoch,
            state.skill.skill_instance_id, state.skill.phase), state)
        state = self.assess(self.observed(state, (0.30, -0.16, 0.02)))
        self.assertEqual(state.phase_feedback.outcome, "passed")
        ended = self.runtime.close_phase(state, self.next_skill)
        self.assertEqual(ended.execution.status, "completed")  # 高度已达不等于 XY 已达。
        self.assertIsNone(self.enter_contact(state).local_intent.handoff)

    def test_stale_partial_and_pre_action_observations_revoke_handoff(self):
        for invalid in ("stale", "partial", "before_action"):
            with self.subTest(invalid=invalid):
                self.runtime.reset()
                # 更换观测前移除旧反馈；enter_contact 会在新观测上重新核验。
                state = replace(self.reached(), phase_feedback=None)
                if invalid == "stale":
                    state = replace(state, freshness=replace(state.freshness, visual_is_fresh=False))
                elif invalid == "partial":
                    state = replace(state, visual_scene=replace(state.visual_scene,
                        entities=(replace(state.visual_scene.entities[0], visibility=Visibility.PARTIAL),)))
                else:
                    state = replace(state, visual_scene=replace(state.visual_scene, capture_timestamp_s=0.49),
                        sources=replace(state.sources, visual_capture_timestamp_s=0.49),
                        freshness=replace(state.freshness, visual_age_s=0.01))
                self.assertIsNone(self.enter_contact(state).local_intent.handoff)

    def test_current_geometry_is_used_even_inside_reference_validity_margin(self):
        state = self.reached()
        entity = state.visual_scene.entities[0]
        footprint = entity.geometry.footprint_xy_robot_base_m.value
        entity = replace(entity, geometry=replace(entity.geometry,
            footprint_xy_robot_base_m=measured(tuple((x, y + 0.012) for x, y in footprint))))
        state = replace(state, visual_scene=replace(state.visual_scene, entities=(entity,)))
        # 12 mm 小于参考失效阈值 15 mm，但大于到达容差 10 mm。
        self.assertIsNone(self.enter_contact(state).local_intent.handoff)

    def test_cumulative_reference_motion_is_checked_against_original_choice(self):
        state = self.enter_contact(self.reached())
        for offset in (0.009, 0.018):
            entity = state.visual_scene.entities[0]
            footprint = entity.geometry.footprint_xy_robot_base_m.value
            entity = replace(entity, geometry=replace(entity.geometry,
                footprint_xy_robot_base_m=measured(tuple((x + 0.009, y) for x, y in footprint))))
            tcp = state.robot.tcp_position_robot_base_m
            state = self.observed(state, (tcp[0] + 0.009, tcp[1], tcp[2]))
            state = replace(state, visual_scene=replace(state.visual_scene, entities=(entity,)))
            state = replace(state, local_intent=self.runtime.context(state))
            self.assertEqual(state.local_intent.handoff is not None, offset < 0.015)

    def test_changed_task_bindings_cannot_inherit_reach(self):
        for field, value in (("subject_ref", "other"), ("skill_instance_id", "other"),
            ("goal", None), ("source_ref", "left_region"), ("target_ref", "table_surface")):
            with self.subTest(field=field):
                self.runtime.reset()
                state = self.reached()
                skill = replace(self.next_skill, **{field: value})
                if field == "subject_ref":
                    state = replace(state, visual_scene=replace(state.visual_scene,
                        entities=(*state.visual_scene.entities, replace(state.visual_scene.entities[0], track_id="other"))))
                state = replace(self.observed(state), skill=skill)
                self.assertIsNone(self.runtime.context(state).handoff)

    def test_pending_feedback_cannot_end_an_intent(self):
        state, _ = self.select()
        self.assertEqual(state.phase_feedback.outcome, "pending")
        with self.assertRaisesRegex(ValueError, "terminal Runtime feedback"):
            self.runtime.close_phase(state, self.next_skill)
        self.assertEqual(self.runtime.context(state).execution.status, "active")

    def test_failed_phase_cancels_intent_without_replan_cost(self):
        state, _ = self.select()
        # 仅验证失败通知的生命周期语义；不是实际失败原因的证据。
        state = replace(state, phase_feedback=replace(state.phase_feedback, outcome="failed"))
        ended = self.runtime.close_phase(state, None)
        self.assertEqual(ended.execution.status, "cancelled")
        self.assertEqual(ended.trigger, "phase_failed")
        self.assertEqual(self.runtime.context(state).replan_count, 0)

    def test_wrong_contract_cannot_complete_the_current_intent(self):
        state, _ = self.select()
        state = replace(state, phase_feedback=replace(state.phase_feedback,
            outcome="passed", contract_id="push_contact_established_v1"))
        with self.assertRaisesRegex(ValueError, "terminal Runtime feedback"):
            self.runtime.close_phase(state, self.next_skill)

    def loop(self):
        # 原共享循环 + 原 verifier/转换；只省去与交接无关的渲染和策略网络。
        loop = object.__new__(BlockPickPlaceRuntimeV1)
        loop._capability_runtime = CapabilityRuntimeV1(self.program, self.assessor.verifier)
        loop._capability_runtime._execution = self.execution
        loop._policy = SimpleNamespace(jev_stages=2)
        loop._local_intent_runtime = self.runtime
        loop._previous_phase_feedback = None
        loop._decision_feedback = None
        loop._pending_xy_alignment_review = None
        loop._events = []
        loop._event_sink = None
        return loop

    def test_shared_loop_preserves_actual_phase_advance_and_records_one_termination(self):
        state, _ = self.select()
        state = self.observed(state, (0.30, -0.16, 0.02))
        loop = self.loop()
        update = loop._capability_runtime.probe_active_phase(state)
        self.assertEqual(update.event_kind, CapabilityRuntimeEventKind.PHASE_ADVANCED)
        self.assertEqual(update.verification.outcome, VerificationOutcome.PASSED)
        result = loop._attach_phase_feedback(state, update)
        self.assertEqual(result.local_intent.execution.status, "superseded")
        loop._attach_phase_feedback(state, update)
        self.assertEqual(sum(e.kind == "local_intent_terminated" for e in loop._events), 1)
        self.assertEqual(self.enter_contact(result).local_intent.last_termination.execution.status, "superseded")

    def test_shared_loop_does_not_end_intent_on_pending_phase(self):
        state, _ = self.select()
        loop = self.loop()
        update = loop._capability_runtime.probe_active_phase(state)
        self.assertEqual(update.event_kind, CapabilityRuntimeEventKind.PHASE_VERIFICATION_PENDING)
        result = loop._attach_phase_feedback(state, update)
        self.assertEqual(result.local_intent.execution.status, "active")
        self.assertFalse(any(e.kind == "local_intent_terminated" for e in loop._events))

    def test_shared_loop_closes_completed_program_without_next_active_skill(self):
        self.state, self.program, self.execution = phase_state(CapabilityKind.PRESS, "press_button")
        state = self.assess(self.state)
        state = replace(state, local_intent=self.runtime.context(state))
        name = state.local_intent.candidates[0].choice_id
        self.runtime.accept(LocalIntentChoiceV1(name, state.state_id, state.action_epoch,
            state.skill.skill_instance_id, state.skill.phase), state)
        from jev4mujoco.contracts.visual_scene import VisualPredicate, VisualRelationV1, EvidenceStatus
        relation = VisualRelationV1(VisualPredicate.PRESSED, state.skill.subject_ref,
            state.skill.target_ref, True, EvidenceStatus.DERIVED, 1.0)
        state = self.observed(state)
        target = state.visual_scene.regions[1]
        target = replace(target, geometry=replace(target.geometry,
            point_robot_base_m=measured(state.robot.tcp_position_robot_base_m)))
        state = replace(state, visual_scene=replace(state.visual_scene,
            regions=(state.visual_scene.regions[0], target), relations=(relation,)),
            physical=replace(state.physical, effector_contact=replace(state.physical.effector_contact,
                mode=EffectorContactMode.PUSH_CONTACT, subject_ref=state.skill.subject_ref,
                left_finger_contact=True)))
        loop = self.loop()
        update = loop._capability_runtime.probe_active_phase(state)
        self.assertEqual(update.event_kind, CapabilityRuntimeEventKind.PROGRAM_COMPLETED)
        self.assertIsNone(update.execution.active_step_index)
        result = loop._attach_phase_feedback(state, update)
        self.assertEqual(result.local_intent.execution.status, "completed")
        self.assertEqual(result.local_intent.last_termination.trigger, "phase_passed")

    def test_final_stop_cancels_active_command_without_claiming_phase_failure(self):
        state, _ = self.select()
        ended = self.runtime.stop(state, "executor timed out")
        self.assertEqual(ended.trigger, "run_stopped")
        self.assertEqual(ended.execution.status, "cancelled")
        self.assertEqual(state.phase_feedback.outcome, "pending")
        self.assertIsNone(self.runtime.stop(state, "executor timed out"))

    def test_pick_and_press_use_same_fresh_point_termination(self):
        for kind, phase, name in ((CapabilityKind.PICK, "align_subject_xy", "align_subject_xy"),
            (CapabilityKind.PRESS, "align_press_point", "align_press_point")):
            with self.subTest(kind=kind):
                self.runtime.reset()
                self.state, self.program, self.execution = phase_state(kind, phase)
                if kind is CapabilityKind.PRESS:
                    target = self.state.visual_scene.regions[1]
                    target = replace(target, geometry=replace(target.geometry,
                        point_robot_base_m=measured((0.42, -0.12, 0.055))))
                    self.state = replace(self.state, visual_scene=replace(self.state.visual_scene,
                        regions=(self.state.visual_scene.regions[0], target)))
                state, chosen = self.select(name)
                state = self.assess(self.observed(state, chosen.candidate.target_robot_base_m))
                self.assertEqual(state.phase_feedback.outcome, "passed")
                ended = self.runtime.close_phase(state, None)
                self.assertEqual(ended.execution.status, "completed")
                self.assertEqual(self.runtime.context(state).replan_count, 0)


if __name__ == "__main__":
    unittest.main()
