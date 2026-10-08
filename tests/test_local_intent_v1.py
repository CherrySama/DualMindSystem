from dataclasses import replace
import unittest

from jev4mujoco.contracts.local_intent import LocalIntentChoiceV1
from jev4mujoco.contracts.visual_scene import Visibility
from jev4mujoco.planning.capabilities import CapabilityKind
from jev4mujoco.policies.decisions import CapabilityDecisionKind
from jev4mujoco.policies.jev import CapabilityJevPolicyV1
from jev4mujoco.runtime.local_intent import (
    LocalIntentRuntimeV1, load_local_intent_config_v1,
)
from tests.test_capability_jev_policy_v1 import FakeJevClient, response
from tests import test_shared_subtask_context as fixtures
from tests.test_shared_subtask_context import ROOT, phase_state
from tests.test_visual_scene_v1 import measured


class LocalIntentV1Test(unittest.TestCase):
    def setUp(self):
        self.assessor = fixtures.SharedSubtaskContextTest()
        self.assessor.setUp()
        self.config = load_local_intent_config_v1(ROOT / "configs/local_intent_v1.yaml")
        self.runtime = LocalIntentRuntimeV1(self.config)
        self.state, self.program, self.execution = phase_state(CapabilityKind.SURFACE_PUSH, "align_precontact")

    def attach(self, state=None):
        state = self.state if state is None else state
        state = replace(state, phase_feedback=None, local_intent=None)
        state = self.assessor.assessed(state, self.program, self.execution)
        return replace(state, local_intent=self.runtime.context(state))

    def choose(self, state, name):
        return LocalIntentChoiceV1(name, state.state_id, state.action_epoch,
            state.skill.skill_instance_id, state.skill.phase)

    def moved(self, state, tcp):
        return replace(state, phase_feedback=None, local_intent=None,
            state_id=state.state_id + 1, action_epoch=state.action_epoch + 1,
            recent_action=replace(state.recent_action, start_action_epoch=state.action_epoch,
                end_action_epoch=state.action_epoch + 1),
            robot=replace(state.robot, tcp_position_robot_base_m=tcp))

    def test_all_observed_polygon_sides_are_choices_not_a_host_selected_route(self):
        state = self.attach()
        sides = [item for item in state.local_intent.candidates if item.choice_id.startswith("approach_edge_")]
        self.assertEqual(len(sides), 4)
        self.assertEqual({item.reference_ref for item in sides}, {state.skill.subject_ref})
        self.assertTrue(all(not item.path_verified for item in sides))
        self.assertTrue(all("up" not in item.allowed_actions and "down" not in item.allowed_actions for item in sides))

    def test_two_calls_bind_actual_selected_intent_and_same_snapshot(self):
        state = self.attach()
        intents = tuple(item.choice_id for item in state.local_intent.candidates) + ("blocked",)
        client = FakeJevClient([response("adjust_push_height", intents), response("down", ("up", "down", "hold", "replan_local_intent", "blocked"))])
        policy = CapabilityJevPolicyV1(client, jev_stages=2)
        decision = policy.decide(state)
        self.assertEqual(decision.action.value, "down")
        self.assertEqual(len(client.requests), 2)
        selected = client.requests[1]["state"]["selected_local_intent"]
        self.assertEqual(selected["choice_id"], "adjust_push_height")
        self.assertAlmostEqual(selected["correction_robot_base_m"][2], -0.05)
        self.assertEqual(client.requests[0]["state"]["state_id"], client.requests[1]["state"]["state_id"])
        self.assertEqual(decision.local_intent_choice.based_on_action_epoch, state.action_epoch)
        motor = client.requests[1]["state"]
        self.assertEqual(motor["schema"], "JEVLocalActionContextV1")
        self.assertNotIn("task", motor)
        self.assertNotIn("subtask_definition", motor)
        self.assertNotIn("candidates", motor["local_intent"])
        self.assertEqual(motor["phase"]["conditions"], client.requests[0]["state"]["phase"]["conditions"])
        self.assertEqual(motor["phase"]["maintain_constraints"], client.requests[0]["state"]["phase"]["maintain_constraints"])
        self.assertEqual(motor["phase"]["verification"]["based_on_action_epoch"], state.action_epoch)
        self.assertEqual(motor["phase"]["binding"]["subject_ref"], state.skill.subject_ref)
        self.assertEqual(selected["axis_error_signs"]["z"], "negative")
        evaluations = {item["action"]: item for item in motor["action_evaluations"]}
        self.assertAlmostEqual(evaluations["down"]["projected_progress_m"], 0.01)
        self.assertAlmostEqual(evaluations["up"]["projected_progress_m"], -0.01)
        self.assertFalse(evaluations["down"]["is_observed"])
        self.assertEqual(evaluations["down"]["source"], "bounded_command_geometry")

    def test_active_intent_is_reused_and_correction_uses_new_observation(self):
        first = self.attach()
        chosen = self.runtime.accept(self.choose(first, "adjust_push_height"), first)
        state = self.attach(self.moved(first, (0.35, 0.0, 0.06)))
        self.assertEqual(state.local_intent.execution.intent_id, chosen.intent_id)
        client = FakeJevClient(response("down", ("up", "down", "hold", "replan_local_intent", "blocked")))
        decision = CapabilityJevPolicyV1(client, jev_stages=2).decide(state)
        self.assertEqual(len(client.requests), 1)
        self.assertAlmostEqual(client.requests[0]["state"]["selected_local_intent"]["correction_robot_base_m"][2], -0.04)
        self.assertEqual(self.runtime.accept(decision.local_intent_choice, state).intent_id, chosen.intent_id)

    def test_local_completion_does_not_complete_capability_phase(self):
        state = self.attach()
        self.runtime.accept(self.choose(state, "adjust_push_height"), state)
        state = self.attach(self.moved(state, (0.35, 0.0, 0.02)))
        self.assertEqual(state.local_intent.execution.status, "completed")
        self.assertEqual(state.phase_feedback.outcome, "pending")
        self.assertTrue(any(item.choice_id.startswith("approach_edge_") for item in state.local_intent.candidates))

    def test_selected_side_is_retained_until_height_adjustment_completes(self):
        state = self.attach()
        execution = self.runtime.accept(self.choose(state, "approach_edge_0"), state)
        target = execution.candidate.target_robot_base_m
        state = self.attach(self.moved(state, target))
        self.assertEqual(state.local_intent.execution.status, "completed")
        self.assertEqual([item.choice_id for item in state.local_intent.candidates], ["adjust_push_height"])
        self.assertEqual(state.local_intent.retained_approach.choice_id, "approach_edge_0")

    def test_grasp_height_intent_requires_existing_xy_alignment_condition(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "descend_to_grasp")
        state = self.assessor.assessed(state, program, execution)
        from jev4mujoco.runtime.local_intent import local_intent_candidates_v1
        candidates = local_intent_candidates_v1(state)
        self.assertEqual([item.choice_id for item in candidates], ["align_grasp_xy"])

    def test_xy_completion_survives_action_permissions_switching_to_height(self):
        self.state, self.program, self.execution = phase_state(CapabilityKind.PICK, "descend_to_grasp")
        state = self.attach()
        selected = self.runtime.accept(self.choose(state, "align_grasp_xy"), state)
        target = selected.candidate.target_robot_base_m
        state = self.moved(state, (target[0], target[1], state.robot.tcp_position_robot_base_m[2]))
        state = self.attach(replace(state, skill=replace(state.skill, allowed_actions=("down",))))
        self.assertEqual(state.local_intent.execution.status, "completed")
        self.assertEqual(state.local_intent.replan_count, 0)
        self.assertEqual([item.choice_id for item in state.local_intent.candidates], ["adjust_grasp_height"])

    def test_executed_xy_feedback_uses_original_point_after_intent_change(self):
        from types import SimpleNamespace
        from jev4mujoco.contracts.state_snapshot import DecisionFeedbackV1
        from jev4mujoco.runtime.loop import BlockPickPlaceRuntimeV1
        state = self.attach()
        selected = self.runtime.accept(self.choose(state, "approach_edge_0"), state)
        state = self.attach(state)
        target = selected.candidate.target_robot_base_m
        moved = self.attach(self.moved(state, (target[0], target[1], target[2])))
        self.runtime.accept(self.choose(moved, "adjust_push_height"), moved)
        moved = replace(moved, local_intent=self.runtime.context(moved))
        review = DecisionFeedbackV1(1, "forward", state.skill.phase, "accepted", "test action", 0.1, 0.09)
        with BlockPickPlaceRuntimeV1(ROOT) as runtime:
            runtime.install_policy(CapabilityJevPolicyV1(FakeJevClient({}), jev_stages=2))
            result = SimpleNamespace(end_action_epoch=moved.action_epoch, end_simulation_time_s=0.0)
            runtime._pending_xy_alignment_review = (review, result, selected.candidate)
            runtime._observe_xy_alignment_progress(moved)
            self.assertIsNone(runtime._progress_block_reason)
            self.assertIsNone(runtime._pending_xy_alignment_review)
            self.assertAlmostEqual(runtime._decision_feedback.error_after_m, 0.0)
            self.assertAlmostEqual(runtime._decision_feedback.progress_m, 0.1)

    def test_oscillation_without_new_best_invalidates_then_requests_local_replanning(self):
        state = self.attach()
        self.runtime.accept(self.choose(state, "adjust_push_height"), state)
        for i in range(self.config.maximum_actions_without_new_best):
            state = self.attach(self.moved(state, (0.35, 0.0, 0.08 if i % 2 == 0 else 0.07)))
        self.assertEqual(state.local_intent.execution.status, "invalidated")
        self.assertIn("no new best", state.local_intent.execution.reason)
        self.assertEqual(state.local_intent.replan_count, 1)
        again = self.attach(state)
        self.assertEqual(again.local_intent.replan_count, 1)

    def test_same_action_epoch_cannot_inflate_nonprogress_count(self):
        state = self.attach()
        self.runtime.accept(self.choose(state, "adjust_push_height"), state)
        for _ in range(10):
            state = self.attach(state)
        self.assertEqual(state.local_intent.execution.nonprogress_actions, 0)

    def test_reference_motion_invalidates_saved_target(self):
        state = self.attach()
        self.runtime.accept(self.choose(state, "approach_edge_0"), state)
        entity = state.visual_scene.entities[0]
        footprint = entity.geometry.footprint_xy_robot_base_m.value
        moved_entity = replace(entity, geometry=replace(entity.geometry,
            footprint_xy_robot_base_m=measured(tuple((x + 0.05, y) for x, y in footprint))))
        state = self.attach(replace(state, visual_scene=replace(state.visual_scene,
            entities=(moved_entity, *state.visual_scene.entities[1:]))))
        self.assertEqual(state.local_intent.execution.status, "invalidated")
        self.assertIn("reference geometry moved", state.local_intent.execution.reason)

    def test_invisible_geometry_is_not_a_zero_error_or_a_completed_intent(self):
        state = self.attach()
        self.runtime.accept(self.choose(state, "approach_edge_0"), state)
        entity = replace(state.visual_scene.entities[0], visibility=Visibility.NOT_DETECTED)
        state = self.attach(replace(state, visual_scene=replace(state.visual_scene,
            entities=(entity, *state.visual_scene.entities[1:]))))
        self.assertEqual(state.local_intent.execution.status, "invalidated")
        self.assertFalse(any(item.choice_id.startswith("approach_edge_") for item in state.local_intent.candidates))

    def test_stale_selection_unknown_candidate_and_silent_intent_switch_are_rejected(self):
        state = self.attach()
        with self.assertRaisesRegex(ValueError, "stale"):
            self.runtime.accept(replace(self.choose(state, "adjust_push_height"), based_on_state_id=1), state)
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.runtime.accept(self.choose(state, "invented_route"), state)
        self.runtime.accept(self.choose(state, "adjust_push_height"), state)
        state = self.attach(state)
        with self.assertRaisesRegex(ValueError, "silently replace"):
            self.runtime.accept(self.choose(state, "approach_edge_0"), state)

    def test_model_blocked_intent_does_not_make_a_motor_request(self):
        state = self.attach()
        choices = tuple(item.choice_id for item in state.local_intent.candidates) + ("blocked",)
        client = FakeJevClient(response("blocked", choices))
        decision = CapabilityJevPolicyV1(client, jev_stages=2).decide(state)
        self.assertEqual(decision.kind, CapabilityDecisionKind.BLOCKED)
        self.assertEqual(len(client.requests), 1)
        self.assertIsNone(decision.action)

    def test_bad_motor_response_preserves_intent_exchange_and_executes_nothing(self):
        state = self.attach()
        choices = tuple(item.choice_id for item in state.local_intent.candidates) + ("blocked",)
        client = FakeJevClient([response("adjust_push_height", choices), {}])
        policy = CapabilityJevPolicyV1(client, jev_stages=2)
        decision = policy.decide(state)
        self.assertEqual(decision.kind, CapabilityDecisionKind.BLOCKED)
        self.assertIsNone(decision.action)
        self.assertIn("intent", policy.last_exchange)
        self.assertIn("error", policy.last_exchange)

    def test_model_can_explicitly_request_revision_without_a_motor_action(self):
        state = self.attach()
        intents = tuple(item.choice_id for item in state.local_intent.candidates) + ("blocked",)
        motors = ("up", "down", "hold", "replan_local_intent", "blocked")
        client = FakeJevClient([response("adjust_push_height", intents), response("replan_local_intent", motors)])
        decision = CapabilityJevPolicyV1(client, jev_stages=2).decide(state)
        self.assertEqual(decision.kind, CapabilityDecisionKind.REPLAN_LOCAL_INTENT)
        self.assertIsNone(decision.action)
        self.runtime.accept(decision.local_intent_choice, state)
        self.runtime.request_replan(decision.reason)
        context = self.runtime.context(state)
        self.assertEqual(context.execution.status, "invalidated")
        self.assertEqual(context.replan_count, 1)

    def test_place_adjustment_and_release_are_distinct_model_choices(self):
        state, program, execution = phase_state(CapabilityKind.PLACE, "lower_to_release")
        state = self.assessor.assessed(state, program, execution)
        from jev4mujoco.runtime.local_intent import local_intent_candidates_v1
        candidates = {item.choice_id: item for item in local_intent_candidates_v1(state)}
        self.assertNotIn("gripper_open", candidates["lower_to_release"].allowed_actions)
        self.assertEqual(candidates["release_subject"].allowed_actions, ("hold", "gripper_open"))

    def test_live_local_replanning_does_not_move_or_reset_the_scene(self):
        from jev4mujoco.runtime.loop import BlockPickPlaceRuntimeV1
        class Client:
            def __init__(self):
                self.queries = 0
            def query(self, request):
                self.queries += 1
                choices = tuple(request["questions"]["decision"]["criteria"])
                choice = "align_subject_xy" if "align_subject_xy" in choices else "replan_local_intent" if self.queries == 2 else "blocked"
                return response(choice, choices)
        client = Client()
        with BlockPickPlaceRuntimeV1(ROOT) as runtime:
            before = runtime.backend.snapshot()
            runtime.install_policy(CapabilityJevPolicyV1(client, jev_stages=2))
            outcome = runtime.run(maximum_decisions=2, maximum_retries=0)
            after = runtime.backend.snapshot()
        self.assertEqual(before.action_epoch, after.action_epoch)
        self.assertEqual(before.simulation_time_s, after.simulation_time_s)
        self.assertFalse(any(event.kind == "action_result" for event in outcome.events))
        self.assertEqual(sum(event.kind == "local_intent_replan_requested" for event in outcome.events), 1)

    def test_unannounced_phase_transition_cancels_plan_without_handoff(self):
        state = self.attach()
        self.runtime.accept(self.choose(state, "approach_edge_0"), state)
        state, self.program, self.execution = phase_state(CapabilityKind.SURFACE_PUSH, "establish_contact")
        state = self.attach(state)
        self.assertIsNone(state.local_intent.execution)
        self.assertIsNone(state.local_intent.retained_approach)
        self.assertIsNone(state.local_intent.handoff)
        self.assertEqual(state.local_intent.last_termination.execution.status, "cancelled")
        self.assertEqual(state.local_intent.last_termination.phase, "align_precontact")
        self.runtime.reset()
        self.assertIsNone(self.attach(state).local_intent.retained_approach)

    def test_local_replanning_is_bounded_without_scene_resets(self):
        state = self.attach()
        for _ in range(self.config.maximum_local_replans):
            self.runtime.accept(self.choose(state, "adjust_push_height"), state)
            for _ in range(self.config.maximum_actions_without_new_best):
                state = self.attach(self.moved(state, (0.35, 0.0, 0.07)))
        self.assertIn("higher-level replanning required", state.local_intent.blocked_reason)

    def test_context_rejects_binding_to_another_snapshot(self):
        state = self.attach()
        with self.assertRaisesRegex(ValueError, "local intent context"):
            replace(state, phase_feedback=None, state_id=state.state_id + 1)

    def test_live_runtime_retains_intent_across_two_actions(self):
        from jev4mujoco.runtime.loop import BlockPickPlaceRuntimeV1
        class Client:
            def __init__(self):
                self.requests = []
            def query(self, request):
                self.requests.append(request)
                choices = tuple(request["questions"]["decision"]["criteria"])
                return response("align_subject_xy" if "align_subject_xy" in choices else "right", choices)
        client = Client()
        with BlockPickPlaceRuntimeV1(ROOT) as runtime:
            runtime.install_policy(CapabilityJevPolicyV1(client, jev_stages=2))
            outcome = runtime.run(maximum_decisions=2, maximum_retries=0)
        self.assertEqual(len(client.requests), 3)  # initial intent + two motor requests
        selections = [event.payload for event in outcome.events if event.kind == "local_intent_selected"]
        continuations = [event.payload for event in outcome.events if event.kind == "local_intent_continued"]
        self.assertEqual(len(selections), 1)
        self.assertEqual(len(continuations), 1)
        self.assertEqual(selections[0].intent_id, continuations[0].intent_id)
        self.assertLess(continuations[0].metric_m, selections[0].metric_m)
        self.assertFalse(outcome.overall_success)  # interface wiring is not task success
        terminal = [event.payload for event in outcome.events if event.kind == "local_intent_terminated"]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].trigger, "run_stopped")
        self.assertEqual(terminal[0].execution.status, "cancelled")
        self.assertEqual(outcome.final_state.local_intent.last_termination, terminal[0])

    def test_selected_push_alignment_reuses_generic_review_without_selecting_an_action(self):
        from jev4mujoco.experiments.registry import EXPERIMENTS_V1
        from jev4mujoco.experiments.scenarios import BlockSurfacePushRuntimeV1
        class Client:
            def __init__(self):
                self.motor_count = 0
                self.requests = []
            def query(self, request):
                self.requests.append(request)
                choices = tuple(request["questions"]["decision"]["criteria"])
                if "prepare_push_effector" in choices:
                    choice = "prepare_push_effector"
                elif "approach_edge_1" in choices:
                    choice = "approach_edge_1"
                elif "gripper_close" in choices:
                    choice = "gripper_close"
                else:
                    self.motor_count += 1
                    choice = ("right", "left", "blocked")[self.motor_count - 1]
                return response(choice, choices)
        client = Client()
        with BlockSurfacePushRuntimeV1(ROOT, **EXPERIMENTS_V1["push_aside"].runtime_kwargs) as runtime:
            runtime.install_policy(CapabilityJevPolicyV1(client, jev_stages=2))
            outcome = runtime.run(maximum_decisions=4, maximum_retries=0)
        reviews = [event.payload for event in outcome.events if event.kind == "decision_review"]
        self.assertEqual([item.status for item in reviews], ["rejected", "accepted", "observed"])
        self.assertEqual([event.payload.action.value for event in outcome.events if event.kind == "action_result"], ["gripper_close", "left"])
        self.assertGreater(reviews[-1].progress_m, 0)
        self.assertEqual(client.requests[-1]["state"]["decision_feedback"]["status"], "observed")
        self.assertFalse(outcome.overall_success)
