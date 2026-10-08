from __future__ import annotations

import unittest
from dataclasses import replace

from jev4mujoco.policies.jev import BLOCKED, JEV_MODEL, CapabilityJevPolicyV1
from jev4mujoco.policies.decisions import CapabilityDecisionKind
from jev4mujoco.policies.decisions import RecoveryChoiceV1
from jev4mujoco.contracts.state_snapshot import DecisionFeedbackV1
from jev4mujoco.policies.typesafe_client import TypeSafeClientError
from tests.test_state_snapshot_v2 import snapshot


def response(choice: str, choices: tuple[str, ...]) -> dict:
    probabilities = {candidate: 0.0 for candidate in choices}
    probabilities[choice] = 1.0
    return {
        "model": JEV_MODEL,
        "answers": {
            "decision": {
                "choice": choice,
                "probabilities": probabilities,
                "confidence": 1.0,
            }
        },
    }


class FakeJevClient:
    def __init__(self, result: object) -> None:
        self.result = result
        self.requests: list[dict] = []
        self.last_call = {"attempts": 1, "http_status": 200, "error": None}

    def query(self, body: dict) -> object:
        self.requests.append(body)
        return self.result.pop(0) if isinstance(self.result, list) else self.result


def motor_snapshot():
    original = snapshot()
    return replace(
        original,
        skill=replace(
            original.skill,
            phase="align_subject_xy",
            allowed_intents=("align_subject_xy",),
            allowed_actions=(
                "forward",
                "backward",
                "left",
                "right",
                "hold",
            ),
        ),
    )


class CapabilityJevPolicyV1Test(unittest.TestCase):
    def test_place_request_exposes_release_gap_and_allows_jev_to_open(self) -> None:
        state = motor_snapshot()
        state = replace(
            state,
            skill=replace(
                state.skill,
                skill_kind="place",
                phase="lower_to_release",
                target_ref="left_region",
                allowed_intents=("lower_to_release", "release_subject"),
                allowed_actions=("down", "up", "hold", "gripper_open"),
            ),
        )
        choices = (*state.skill.allowed_actions, BLOCKED)
        client = FakeJevClient(response("gripper_open", choices))
        decision = CapabilityJevPolicyV1(client).decide(state)

        geometry = client.requests[0]["state"]["interaction"]["release_geometry"]
        self.assertEqual(decision.action.value, "gripper_open")
        self.assertTrue(geometry["observable"])
        self.assertAlmostEqual(geometry["tcp_to_target_support_vertical_m"], 0.07)
        self.assertAlmostEqual(geometry["subject_bottom_to_target_support_clearance_m"], 0.0)

    def test_jev_can_choose_retry_from_recovery_state(self) -> None:
        state = motor_snapshot()
        choices = ("observe", "retry_pick", "stop")
        client = FakeJevClient(response("retry_pick", choices))
        decision = CapabilityJevPolicyV1(client).decide_recovery(
            state, "subject was lost during transport"
        )

        self.assertEqual(decision.choice, RecoveryChoiceV1.RETRY_PICK)
        self.assertEqual(decision.based_on_state_id, state.state_id)
        self.assertEqual(
            client.requests[0]["state"]["recovery"]["failure_reason"],
            "subject was lost during transport",
        )

    def test_transport_failure_records_client_diagnostic_without_action(self) -> None:
        class FailingClient:
            last_call = {"attempts": 3, "error": "transport_error"}

            def query(self, body):
                raise TypeSafeClientError("TypeSafe transport error; no action executed")

        exchanges = []
        decision = CapabilityJevPolicyV1(
            FailingClient(), exchange_sink=exchanges.append
        ).decide(motor_snapshot())

        self.assertEqual(decision.kind, CapabilityDecisionKind.BLOCKED)
        self.assertEqual(exchanges[0]["payload"]["transport"]["attempts"], 3)

    def test_inconsistent_choice_retries_same_snapshot_then_uses_jev_choice(self) -> None:
        state = motor_snapshot()
        choices = (*state.skill.allowed_actions, BLOCKED)
        inconsistent = response("left", choices)
        inconsistent["answers"]["decision"]["probabilities"] = {
            "forward": 0.0, "backward": 0.0, "left": 0.4,
            "right": 0.6, "hold": 0.0, BLOCKED: 0.0,
        }
        client = FakeJevClient([inconsistent, response("right", choices)])
        exchanges = []
        decision = CapabilityJevPolicyV1(client, exchange_sink=exchanges.append).decide(state)

        self.assertEqual(decision.action.value, "right")
        self.assertEqual(client.requests[0], client.requests[1])
        self.assertEqual(len(exchanges[0]["payload"]["attempts"]), 2)

    def test_inconsistent_choice_twice_blocks_without_action(self) -> None:
        state = motor_snapshot()
        choices = (*state.skill.allowed_actions, BLOCKED)
        inconsistent = response("left", choices)
        inconsistent["answers"]["decision"]["probabilities"] = {
            "forward": 0.0, "backward": 0.0, "left": 0.4,
            "right": 0.6, "hold": 0.0, BLOCKED: 0.0,
        }
        client = FakeJevClient([inconsistent, inconsistent])
        decision = CapabilityJevPolicyV1(client).decide(state)

        self.assertEqual(decision.kind, CapabilityDecisionKind.BLOCKED)
        self.assertIsNone(decision.action)
        self.assertEqual(len(client.requests), 2)

    def test_valid_choice_becomes_one_atomic_action(self) -> None:
        choices = (
            "forward",
            "backward",
            "left",
            "right",
            "hold",
            BLOCKED,
        )
        client = FakeJevClient(response("forward", choices))
        policy = CapabilityJevPolicyV1(client)

        decision = policy.decide(motor_snapshot())

        self.assertEqual(decision.kind, CapabilityDecisionKind.ACTION)
        self.assertEqual(decision.action.value, "forward")
        self.assertEqual(len(client.requests), 1)
        request = client.requests[0]
        self.assertEqual(
            request["state"]["schema"], "JEVDecisionContextV1"
        )
        self.assertEqual(
            request["state"]["handle"]["translation_step_m"], 0.01
        )
        self.assertEqual(request["state"]["interaction"]["active_axes"], ["x", "y"])
        self.assertNotIn("up", request["questions"]["decision"]["criteria"])
        self.assertEqual(
            tuple(request["questions"]["decision"]["criteria"]), choices
        )

    def test_verification_only_phase_skips_remote_call(self) -> None:
        state = motor_snapshot()
        state = replace(
            state,
            skill=replace(
                state.skill,
                phase="verify_grasp",
                allowed_intents=("request_grasp_verification",),
                allowed_actions=(),
            ),
        )
        client = FakeJevClient({})

        decision = CapabilityJevPolicyV1(client).decide(state)

        self.assertEqual(decision.kind, CapabilityDecisionKind.BLOCKED)
        self.assertIn("Runtime must handle", decision.reason)
        self.assertEqual(client.requests, [])

    def test_rejected_direction_is_visible_to_next_jev_request(self) -> None:
        state = replace(
            motor_snapshot(),
            decision_feedback=DecisionFeedbackV1(
                decision_id=1,
                action="left",
                phase="align_subject_xy",
                status="rejected",
                reason="action does not reduce projected pick XY error enough",
                error_before_m=0.1818,
                projected_error_after_m=0.1917,
            ),
        )
        choices = (*state.skill.allowed_actions, BLOCKED)
        client = FakeJevClient(response("right", choices))

        decision = CapabilityJevPolicyV1(client).decide(state)

        self.assertEqual(decision.action.value, "right")
        feedback = client.requests[0]["state"]["decision_feedback"]
        self.assertEqual(feedback["action"], "left")
        self.assertEqual(feedback["status"], "rejected")
        self.assertGreater(
            feedback["projected_error_after_m"], feedback["error_before_m"]
        )

    def test_malformed_or_disallowed_response_blocks_without_action(self) -> None:
        choices = (
            "forward",
            "backward",
            "left",
            "right",
            "hold",
            BLOCKED,
        )
        malformed = response("forward", choices)
        malformed["model"] = "unexpected-model"
        exchanges = []
        policy = CapabilityJevPolicyV1(
            FakeJevClient(malformed), exchange_sink=exchanges.append
        )

        decision = policy.decide(motor_snapshot())

        self.assertEqual(decision.kind, CapabilityDecisionKind.BLOCKED)
        self.assertIsNone(decision.action)
        self.assertIn("malformed", decision.reason)
        self.assertEqual(exchanges[0]["kind"], "jev_exchange")
        self.assertEqual(
            exchanges[0]["payload"]["response"]["model"],
            "unexpected-model",
        )

        disallowed = response("rotate_cw", (*choices, "rotate_cw"))
        decision = CapabilityJevPolicyV1(FakeJevClient(disallowed)).decide(
            motor_snapshot()
        )
        self.assertEqual(decision.kind, CapabilityDecisionKind.BLOCKED)
        self.assertIsNone(decision.action)


if __name__ == "__main__":
    unittest.main()
