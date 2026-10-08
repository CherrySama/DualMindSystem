from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from jev4mujoco.runtime.capability_runtime import (
    CapabilityRuntimeEventKind,
    CapabilityRuntimeV1,
    PhaseVerificationRequestV1,
)
from jev4mujoco.planning.capabilities import CapabilityProgramStatus
from jev4mujoco.planning.compiler import compile_pick_and_place_capabilities_v1
from jev4mujoco.runtime.verifier import RuntimeVerifierV1
from jev4mujoco.runtime.verifier_config import load_runtime_verifier_config_v1
from tests.test_capability_v1 import pick_and_place_plan
from tests.test_runtime_verifier_v1 import snapshot_for_execution
from tests.test_state_snapshot_v2 import snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fresh_entry_snapshot():
    original = snapshot()
    visual = replace(
        original.visual_scene,
        observation_id=2,
        capture_timestamp_s=0.6,
        publish_timestamp_s=0.6,
    )
    return replace(
        original,
        timestamp_s=0.6,
        sources=replace(
            original.sources,
            visual_observation_id=2,
            visual_capture_timestamp_s=0.6,
            robot_measurement_timestamp_s=0.6,
        ),
        visual_scene=visual,
        freshness=replace(
            original.freshness,
            visual_age_s=0.0,
            visual_is_fresh=True,
        ),
    )


class CapabilityRuntimeV1Test(unittest.TestCase):
    def setUp(self) -> None:
        plan, initial_scene = pick_and_place_plan()
        self.program = compile_pick_and_place_capabilities_v1(plan, initial_scene)
        config = load_runtime_verifier_config_v1(
            PROJECT_ROOT / "configs/runtime_verifier_v1.yaml"
        )
        self.runtime = CapabilityRuntimeV1(
            self.program, RuntimeVerifierV1(config)
        )

    def test_entry_contract_activates_first_capability(self) -> None:
        update = self.runtime.observe(fresh_entry_snapshot())
        self.assertEqual(
            update.event_kind,
            CapabilityRuntimeEventKind.CAPABILITY_ACTIVATED,
        )
        self.assertEqual(self.runtime.execution.active_step_index, 0)
        self.assertEqual(
            self.runtime.execution.status,
            CapabilityProgramStatus.RUNNING,
        )

    def test_entry_pending_does_not_mutate_execution(self) -> None:
        current = fresh_entry_snapshot()
        current = replace(
            current,
            physical=replace(
                current.physical,
                support_state=replace(
                    current.physical.support_state,
                    supported=False,
                ),
            ),
        )
        before = self.runtime.execution
        update = self.runtime.observe(current)
        self.assertEqual(update.event_kind, CapabilityRuntimeEventKind.ENTRY_PENDING)
        self.assertIs(self.runtime.execution, before)

    def test_stale_or_wrong_skill_request_cannot_change_phase(self) -> None:
        self.runtime.observe(fresh_entry_snapshot())
        current = snapshot_for_execution(
            self.program,
            self.runtime.execution,
            observation_id=3,
            timestamp_s=0.7,
        )
        before = self.runtime.execution
        with self.assertRaisesRegex(ValueError, "stale"):
            self.runtime.request_phase_verification(
                PhaseVerificationRequestV1(
                    request_id=1,
                    skill_instance_id=self.program.steps[0].instance_id,
                    based_on_state_id=current.state_id + 1,
                    based_on_action_epoch=current.action_epoch,
                ),
                current,
            )
        self.assertIs(self.runtime.execution, before)

        with self.assertRaisesRegex(ValueError, "different skill"):
            self.runtime.request_phase_verification(
                PhaseVerificationRequestV1(
                    request_id=2,
                    skill_instance_id=self.program.steps[1].instance_id,
                    based_on_state_id=current.state_id,
                    based_on_action_epoch=current.action_epoch,
                ),
                current,
            )
        self.assertIs(self.runtime.execution, before)

    def test_runtime_not_policy_advances_verified_phase(self) -> None:
        self.runtime.observe(fresh_entry_snapshot())
        current = snapshot_for_execution(
            self.program,
            self.runtime.execution,
            observation_id=3,
            timestamp_s=0.7,
        )
        request_update = self.runtime.request_phase_verification(
            PhaseVerificationRequestV1(
                request_id=1,
                skill_instance_id=self.program.steps[0].instance_id,
                based_on_state_id=current.state_id,
                based_on_action_epoch=current.action_epoch,
            ),
            current,
        )
        self.assertEqual(
            request_update.event_kind,
            CapabilityRuntimeEventKind.PHASE_VERIFICATION_REQUESTED,
        )

        awaiting_snapshot = snapshot_for_execution(
            self.program,
            self.runtime.execution,
            observation_id=4,
            timestamp_s=0.8,
        )
        verification_update = self.runtime.observe(awaiting_snapshot)
        self.assertEqual(
            verification_update.event_kind,
            CapabilityRuntimeEventKind.PHASE_ADVANCED,
        )
        self.assertEqual(
            self.runtime.execution.steps[0].phase_index,
            1,
        )

    def test_automatic_probe_advances_a_completed_phase(self) -> None:
        self.runtime.observe(fresh_entry_snapshot())
        current = snapshot_for_execution(
            self.program,
            self.runtime.execution,
            observation_id=3,
            timestamp_s=0.7,
        )

        update = self.runtime.probe_active_phase(current)

        self.assertEqual(
            update.event_kind,
            CapabilityRuntimeEventKind.PHASE_ADVANCED,
        )
        self.assertEqual(self.runtime.execution.steps[0].phase_index, 1)

    def test_pending_automatic_probe_leaves_phase_running(self) -> None:
        self.runtime.observe(fresh_entry_snapshot())
        current = snapshot_for_execution(
            self.program,
            self.runtime.execution,
            observation_id=3,
            timestamp_s=0.7,
        )
        current = replace(
            current,
            robot=replace(
                current.robot,
                gripper=replace(
                    current.robot.gripper,
                    opening_m=0.0,
                    target_opening_m=0.0,
                ),
            ),
        )
        before = self.runtime.execution

        update = self.runtime.probe_active_phase(current)

        self.assertEqual(
            update.event_kind,
            CapabilityRuntimeEventKind.PHASE_VERIFICATION_PENDING,
        )
        self.assertIs(self.runtime.execution, before)
        self.assertEqual(self.runtime.execution.steps[0].status.value, "running")


if __name__ == "__main__":
    unittest.main()
