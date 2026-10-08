from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from jev4mujoco.contracts.actions import AtomicAction, ActionStatus
from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1,
    CapabilityExecutionStateV1,
    CapabilityProgramStatus,
    CapabilityRunStatus,
    CapabilityStepStateV1,
    VerificationOutcome,
)
from jev4mujoco.experiments.scenarios import (
    BlockPlaceInBowlRuntimeV1, MultiBlockStackRuntimeV1, BlockSurfacePushRuntimeV1,
    LargeButtonPressRuntimeV1, MultiBlockSortRuntimeV1,
)
from jev4mujoco.runtime.loop import BlockPickPlaceRuntimeV1
from jev4mujoco.policies.decisions import (
    CapabilityDecisionKind,
    CapabilityDecisionV1,
    RecoveryChoiceV1,
    RecoveryDecisionV1,
)
from jev4mujoco.runtime.verifier import VerificationResultV1


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BlockRuntimeV1Test(unittest.TestCase):
    def test_atomic_timeout_enters_recovery_without_waiting_for_decision_budget(self) -> None:
        class TimeoutPolicy:
            calls = 0

            def decide(self, state):
                self.calls += 1
                return CapabilityDecisionV1(
                    self.calls, CapabilityDecisionKind.ACTION, state.state_id,
                    state.action_epoch, state.skill.skill_instance_id,
                    AtomicAction(state.skill.allowed_actions[0]), "injected timeout action",
                )

            def decide_recovery(self, state, reason):
                return RecoveryDecisionV1(RecoveryChoiceV1.RETRY_CAPABILITY,
                                          state.state_id, state.action_epoch, reason)

        with BlockSurfacePushRuntimeV1(PROJECT_ROOT) as runtime:
            policy = TimeoutPolicy()
            runtime.install_policy(policy)
            execute = runtime._execute

            def timed_out(decision, state):
                return replace(execute(decision, state), status=ActionStatus.TIMED_OUT,
                               reason="injected atomic timeout")

            runtime._execute = timed_out
            outcome = runtime.run(maximum_decisions=140, maximum_retries=5)
            self.assertEqual(policy.calls, 6)
            self.assertEqual(runtime._retry_count, 5)
            self.assertEqual(outcome.decision_count, 6)
            self.assertIn("maximum retry budget exhausted (5)", outcome.reason)
            self.assertIn("injected atomic timeout", outcome.reason)

    def test_five_retries_preserve_scene_and_stop_after_six_failures(self) -> None:
        class AlwaysBlockedPolicy:
            calls = 0

            def decide(self, state):
                self.calls += 1
                return CapabilityDecisionV1(
                    self.calls, CapabilityDecisionKind.BLOCKED, state.state_id,
                    state.action_epoch, state.skill.skill_instance_id, None,
                    "injected execution failure",
                )

            def decide_recovery(self, state, reason):
                return RecoveryDecisionV1(
                    RecoveryChoiceV1.RETRY_CAPABILITY, state.state_id,
                    state.action_epoch, reason,
                )

        for runtime_type in (BlockPickPlaceRuntimeV1, BlockSurfacePushRuntimeV1, LargeButtonPressRuntimeV1):
            with self.subTest(runtime=runtime_type.__name__), runtime_type(PROJECT_ROOT) as runtime:
                policy = AlwaysBlockedPolicy()
                runtime.install_policy(policy)
                initial = runtime.backend.snapshot()
                outcome = runtime.run(maximum_decisions=20, maximum_retries=5)
                resets = [event for event in outcome.events if event.kind == "capability_runtime"
                          and event.payload.event_kind.value == "recovery_reset"]
                self.assertEqual(len(resets), 5)
                self.assertEqual(policy.calls, 6)
                self.assertIn("maximum retry budget exhausted (5)", outcome.reason)
                self.assertEqual(initial.bowl_position_robot_base_m, outcome.final_raw_state.bowl_position_robot_base_m)
                self.assertEqual(initial.action_epoch, outcome.final_raw_state.action_epoch)
                self.assertFalse(outcome.overall_success)

    def test_retry_of_second_object_preserves_completed_first_object(self) -> None:
        with MultiBlockSortRuntimeV1(PROJECT_ROOT) as runtime:
            program = runtime.program
            runtime._capability_runtime._execution = CapabilityExecutionStateV1(
                program_id=program.program_id, status=CapabilityProgramStatus.BLOCKED,
                active_step_index=4, transition_index=12, blocked_reason="second object dropped",
                steps=tuple(CapabilityStepStateV1(
                    instance_id=step.instance_id,
                    status=(CapabilityRunStatus.COMPLETED if index < 4 else
                            CapabilityRunStatus.BLOCKED if index == 4 else CapabilityRunStatus.PENDING),
                    phase_index=0,
                ) for index, step in enumerate(program.steps)),
            )
            before = runtime._capability_runtime.execution.steps[:3]
            runtime._capability_runtime.restart_current_capability(runtime._fresh_state())
            self.assertEqual(before, runtime._capability_runtime.execution.steps[:3])
            self.assertTrue(all(step.status is CapabilityRunStatus.PENDING
                                for step in runtime._capability_runtime.execution.steps[3:]))

    def test_decision_budget_is_renewed_only_after_actual_retry(self) -> None:
        class PrematureVerificationPolicy:
            calls = 0

            def decide(self, state):
                self.calls += 1
                return CapabilityDecisionV1(
                    self.calls, CapabilityDecisionKind.REQUEST_VERIFICATION,
                    state.state_id, state.action_epoch, state.skill.skill_instance_id,
                    None, "injected premature verification",
                )

            def decide_recovery(self, state, reason):
                return RecoveryDecisionV1(RecoveryChoiceV1.RETRY_PICK, state.state_id,
                                          state.action_epoch, reason)

        with BlockPickPlaceRuntimeV1(PROJECT_ROOT) as runtime:
            policy = PrematureVerificationPolicy()
            runtime.install_policy(policy)
            outcome = runtime.run(maximum_decisions=1, maximum_retries=5)
            self.assertEqual(policy.calls, 6)
            self.assertEqual(runtime._retry_count, 5)
            self.assertIn("maximum decision budget exhausted", outcome.reason)

    def test_runtime_requires_policy_before_progress_or_actions(self) -> None:
        with BlockPickPlaceRuntimeV1(PROJECT_ROOT) as runtime:
            execution = runtime._capability_runtime.execution
            before = runtime.backend.snapshot()
            observations = runtime._observation_id
            with self.assertRaisesRegex(RuntimeError, "install a policy"):
                runtime.run(maximum_decisions=1)
            self.assertIs(runtime._capability_runtime.execution, execution)
            self.assertEqual(runtime.backend.snapshot().action_epoch, before.action_epoch)
            self.assertEqual(runtime._observation_id, observations)
            self.assertFalse(runtime._events)

    def test_bowl_containment_uses_rotated_block_corners(self) -> None:
        with BlockPlaceInBowlRuntimeV1(PROJECT_ROOT) as runtime:
            initial = runtime.backend.snapshot()
            bowl_center = initial.target_position_robot_base_m
            raw = replace(
                initial,
                bowl_position_robot_base_m=(
                    bowl_center[0] - 0.01070583134025967,
                    bowl_center[1] - 0.03050724796845393,
                    0.027383250101066466,
                ),
                bowl_orientation_robot_base_wxyz=(
                    0.6985086093393358, 0.013998505309594407,
                    0.06603884082526915, 0.7124104407084298,
                ),
            )
            result = runtime._hidden_truth(raw)

        self.assertTrue(result.fully_inside)

    def test_retry_restarts_current_stack_chain_without_resetting_scene(self) -> None:
        class RetryPolicy:
            def decide_recovery(self, state, failure_reason):
                return RecoveryDecisionV1(
                    choice=RecoveryChoiceV1.RETRY_PICK,
                    based_on_state_id=state.state_id,
                    based_on_action_epoch=state.action_epoch,
                    reason=f"retry after {failure_reason}",
                )

        with MultiBlockStackRuntimeV1(PROJECT_ROOT) as runtime:
            runtime.install_policy(RetryPolicy())
            first_position = runtime.backend.snapshot().bowl_position_robot_base_m
            program = runtime.program
            runtime._capability_runtime._execution = CapabilityExecutionStateV1(
                program_id=program.program_id,
                status=CapabilityProgramStatus.BLOCKED,
                active_step_index=1,
                steps=tuple(
                    CapabilityStepStateV1(
                        instance_id=step.instance_id,
                        status=(CapabilityRunStatus.COMPLETED if index == 0
                                else CapabilityRunStatus.BLOCKED if index == 1
                                else CapabilityRunStatus.PENDING),
                        phase_index=0,
                    )
                    for index, step in enumerate(program.steps)
                ),
                transition_index=2,
                blocked_reason="subject was lost during transport",
            )
            state = runtime._fresh_state()
            self.assertTrue(state.physical.support_state.supported)
            self.assertTrue(runtime._attempt_recovery(state, "subject was lost during transport"))
            self.assertTrue(any(event.kind == "recovery_decision" for event in runtime._events))
            self.assertEqual(
                runtime.backend.snapshot().bowl_position_robot_base_m,
                first_position,
            )
            execution = runtime._capability_runtime.execution
            self.assertEqual(execution.status, CapabilityProgramStatus.RUNNING)
            self.assertIsNone(execution.active_step_index)
            self.assertTrue(all(step.status is CapabilityRunStatus.PENDING for step in execution.steps))
            resets = [event for event in runtime._events if event.kind == "capability_runtime"
                      and event.payload.event_kind.value == "recovery_reset"]
            self.assertEqual(len(resets), 1)
            self.assertFalse(any(event.kind == "action_result" for event in runtime._events))

        self.assertTrue(runtime.program.steps[0].instance_id.endswith("retry3"))

    def test_pending_placement_verification_has_finite_timeout(self) -> None:
        with MultiBlockStackRuntimeV1(PROJECT_ROOT) as runtime:
            runtime._verifier_config = replace(
                runtime._verifier_config, maximum_pending_verification_s=0.6
            )
            class VerificationOnlyPolicy:
                def decide(self, state):
                    raise AssertionError("verification-only phase must not call policy")

            runtime.install_policy(VerificationOnlyPolicy())
            program = runtime.program
            placement = program.steps[2]
            phase_index = next(index for index, phase in enumerate(
                CAPABILITY_TEMPLATES_V1[placement.kind].phases
            ) if phase.phase_id == "verify_placement")
            runtime._capability_runtime._execution = CapabilityExecutionStateV1(
                program_id=program.program_id,
                status=CapabilityProgramStatus.RUNNING,
                active_step_index=2,
                steps=tuple(CapabilityStepStateV1(
                    instance_id=step.instance_id,
                    status=CapabilityRunStatus.COMPLETED if index < 2 else CapabilityRunStatus.RUNNING,
                    phase_index=phase_index if index == 2 else 0,
                ) for index, step in enumerate(program.steps)),
                transition_index=2,
            )
            verifier = runtime._capability_runtime._verifier
            verifier._verify_place_goal_stable_v1 = lambda instance, state: VerificationResultV1(
                contract_id="place_goal_stable_v1",
                outcome=VerificationOutcome.PENDING,
                reason="forced pending placement",
                based_on_observation_id=state.visual_scene.observation_id,
                based_on_state_id=state.state_id,
                based_on_action_epoch=state.action_epoch,
                based_on_skill=state.skill,
            )
            outcome = runtime.run(maximum_decisions=160)

        self.assertEqual(outcome.reason, "verify_placement verification timed out")
        self.assertFalse(outcome.overall_success)

    def test_pick_xy_rejects_wrong_direction_then_observes_correction(self) -> None:
        class TwoChoicePolicy:
            def __init__(self) -> None:
                self.states = []

            def decide(self, state):
                self.states.append(state)
                choice = AtomicAction.LEFT if len(self.states) == 1 else AtomicAction.RIGHT
                return CapabilityDecisionV1(
                    decision_id=len(self.states),
                    kind=CapabilityDecisionKind.ACTION,
                    based_on_state_id=state.state_id,
                    based_on_action_epoch=state.action_epoch,
                    skill_instance_id=state.skill.skill_instance_id,
                    action=choice,
                    reason="replay the initial wrong direction and its correction",
                )

        policy = TwoChoicePolicy()
        with MultiBlockStackRuntimeV1(PROJECT_ROOT) as runtime:
            runtime.install_policy(policy)
            outcome = runtime.run(maximum_decisions=2)

        reviews = [event.payload for event in outcome.events if event.kind == "decision_review"]
        actions = [event.payload for event in outcome.events if event.kind == "action_result"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, AtomicAction.RIGHT)
        self.assertEqual([review.status for review in reviews], ["rejected", "accepted", "observed"])
        self.assertEqual(policy.states[1].decision_feedback.status, "rejected")
        self.assertEqual(policy.states[0].action_epoch, policy.states[1].action_epoch)
        self.assertGreater(reviews[2].progress_m, 0.0)

    def test_pick_xy_stops_repeated_wrong_direction_without_motion(self) -> None:
        class WrongDirectionPolicy:
            def __init__(self) -> None:
                self.calls = 0

            def decide(self, state):
                self.calls += 1
                return CapabilityDecisionV1(
                    decision_id=self.calls,
                    kind=CapabilityDecisionKind.ACTION,
                    based_on_state_id=state.state_id,
                    based_on_action_epoch=state.action_epoch,
                    skill_instance_id=state.skill.skill_instance_id,
                    action=AtomicAction.LEFT,
                    reason="repeat the failed JEV choice",
                )

        with MultiBlockStackRuntimeV1(PROJECT_ROOT) as runtime:
            runtime.install_policy(WrongDirectionPolicy())
            outcome = runtime.run(maximum_decisions=5)

        self.assertIn("rejected repeatedly", outcome.reason)
        self.assertEqual(outcome.decision_count, 2)
        self.assertFalse(any(event.kind == "action_result" for event in outcome.events))

    def test_runtime_accepts_an_injected_capability_policy(self) -> None:
        class BlockingPolicy:
            def __init__(self) -> None:
                self.calls = 0
                self.phases = []

            def decide(self, snapshot):
                self.calls += 1
                self.phases.append(snapshot.skill.phase)
                return CapabilityDecisionV1(
                    decision_id=self.calls,
                    kind=CapabilityDecisionKind.BLOCKED,
                    based_on_state_id=snapshot.state_id,
                    based_on_action_epoch=snapshot.action_epoch,
                    skill_instance_id=snapshot.skill.skill_instance_id,
                    action=None,
                    reason="injected policy stopped the run",
                )

        policy = BlockingPolicy()
        with BlockPickPlaceRuntimeV1(PROJECT_ROOT) as runtime:
            runtime.install_policy(policy)
            outcome = runtime.run(maximum_decisions=1)

        self.assertTrue(outcome.blocked)
        self.assertEqual(outcome.reason, "injected policy stopped the run")
        self.assertEqual(policy.calls, 1)
        self.assertEqual(policy.phases, ["align_subject_xy"])


if __name__ == "__main__":
    unittest.main()
