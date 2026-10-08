from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1,
    CapabilityExecutionStateV1,
    CapabilityKind,
    VerificationOutcome,
)
from jev4mujoco.runtime.transitions import (
    activate_next_capability_v1,
    active_skill_state_v2,
    apply_active_verification_v1,
    initial_capability_execution_state_v1,
    request_active_phase_verification_v1,
    start_capability_program_v1,
)
from jev4mujoco.planning.compiler import (
    compile_pick_and_place_capabilities_v1,
    compile_surface_push_capabilities_v1,
)
from jev4mujoco.runtime.verifier import RuntimeVerifierV1
from jev4mujoco.runtime.verifier_config import load_runtime_verifier_config_v1
from jev4mujoco.contracts.state_snapshot import (
    GraspState,
    PhysicalEvidenceV2,
    SkillRunStatus,
)
from jev4mujoco.contracts.task_plan import (
    PushToRegionTaskV1,
    RegionRefV1,
    make_task_plan_v1,
)
from jev4mujoco.contracts.visual_scene import (
    EvidenceStatus,
    RegionType,
    VisualPredicate,
    VisualRelationV1,
)
from tests.test_capability_v1 import pick_and_place_plan
from tests.test_state_snapshot_v2 import snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/runtime_verifier_v1.yaml"


def execution_at_place_verification():
    plan, initial_scene = pick_and_place_plan()
    program = compile_pick_and_place_capabilities_v1(plan, initial_scene)
    execution = start_capability_program_v1(
        program, initial_capability_execution_state_v1(program)
    )
    for kind in (CapabilityKind.PICK, CapabilityKind.TRANSPORT):
        for _ in CAPABILITY_TEMPLATES_V1[kind].phases:
            execution = request_active_phase_verification_v1(program, execution)
            execution = apply_active_verification_v1(
                program, execution, VerificationOutcome.PASSED
            )
        execution = activate_next_capability_v1(program, execution)
    for _ in range(
        len(CAPABILITY_TEMPLATES_V1[CapabilityKind.PLACE].phases) - 1
    ):
        execution = request_active_phase_verification_v1(program, execution)
        execution = apply_active_verification_v1(
            program, execution, VerificationOutcome.PASSED
        )
    execution = request_active_phase_verification_v1(program, execution)
    return program, execution


def snapshot_for_execution(
    program,
    execution: CapabilityExecutionStateV1,
    *,
    observation_id: int,
    timestamp_s: float,
    in_target: bool = False,
):
    original = snapshot()
    entity = original.visual_scene.entities[0]
    if in_target:
        geometry = replace(
            entity.geometry,
            centroid_robot_base_m=replace(
                entity.geometry.centroid_robot_base_m,
                value=(0.275, 0.125, 0.03),
            ),
            top_center_robot_base_m=replace(
                entity.geometry.top_center_robot_base_m,
                value=(0.275, 0.125, 0.06),
            ),
            footprint_xy_robot_base_m=replace(
                entity.geometry.footprint_xy_robot_base_m,
                value=(
                    (0.245, 0.095),
                    (0.305, 0.095),
                    (0.305, 0.155),
                    (0.245, 0.155),
                ),
            ),
        )
        entity = replace(entity, geometry=geometry, support_region_ref="table_surface")
    relations = tuple(original.visual_scene.relations)
    if in_target:
        relations += (
            VisualRelationV1(
                predicate=VisualPredicate.IN_REGION,
                subject_ref="track_001",
                reference_ref="left_region",
                value=True,
                status=EvidenceStatus.DERIVED,
                confidence=1.0,
            ),
        )
    visual = replace(
        original.visual_scene,
        observation_id=observation_id,
        capture_timestamp_s=timestamp_s,
        publish_timestamp_s=timestamp_s,
        entities=(entity,),
        relations=relations,
    )
    return replace(
        original,
        timestamp_s=timestamp_s,
        sources=replace(
            original.sources,
            visual_observation_id=observation_id,
            visual_capture_timestamp_s=timestamp_s,
            robot_measurement_timestamp_s=timestamp_s,
        ),
        skill=active_skill_state_v2(program, execution),
        visual_scene=visual,
        freshness=replace(
            original.freshness,
            visual_age_s=0.0,
            visual_is_fresh=True,
        ),
    )


class RuntimeVerifierV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_runtime_verifier_config_v1(CONFIG_PATH)
        self.verifier = RuntimeVerifierV1(self.config)
        self.plan, self.initial_scene = pick_and_place_plan()
        self.program = compile_pick_and_place_capabilities_v1(
            self.plan, self.initial_scene
        )

    def test_every_template_contract_has_a_verifier_route(self) -> None:
        self.assertEqual(
            {
                template.entry_contract
                for template in CAPABILITY_TEMPLATES_V1.values()
                if template.runtime_implemented
            },
            {
                "subject_visible_and_supported_v1",
                "subject_held_v1",
                "subject_held_above_region_v1",
                "subject_visible_supported_on_surface_v1",
                "button_visible_v1",
            },
        )
        for template in CAPABILITY_TEMPLATES_V1.values():
            if not template.runtime_implemented:
                continue
            for phase in template.phases:
                self.assertTrue(
                    hasattr(
                        self.verifier,
                        f"_verify_{phase.completion_contract}",
                    ),
                    phase.completion_contract,
                )

    def test_pick_entry_and_open_gripper_phase_pass_from_public_state(self) -> None:
        execution = start_capability_program_v1(
            self.program, initial_capability_execution_state_v1(self.program)
        )
        current = snapshot_for_execution(
            self.program,
            execution,
            observation_id=2,
            timestamp_s=0.6,
        )
        entry = self.verifier.verify_entry(self.program.steps[0], current)
        self.assertEqual(entry.outcome, VerificationOutcome.PASSED)

        execution = request_active_phase_verification_v1(self.program, execution)
        current = replace(
            current,
            skill=active_skill_state_v2(self.program, execution),
        )
        result = self.verifier.verify_active_phase(
            self.program, execution, current
        )
        self.assertEqual(result.contract_id, "gripper_open_v1")
        self.assertEqual(result.outcome, VerificationOutcome.PASSED)

    def test_transport_entry_does_not_pass_until_subject_is_held(self) -> None:
        instance = self.program.steps[1]
        execution = start_capability_program_v1(
            self.program, initial_capability_execution_state_v1(self.program)
        )
        current = snapshot_for_execution(
            self.program,
            execution,
            observation_id=2,
            timestamp_s=0.6,
        )
        waiting = self.verifier.verify_entry(instance, current)
        self.assertEqual(waiting.outcome, VerificationOutcome.PENDING)

        held_physical = replace(
            current.physical,
            grasp_state=GraspState.HELD,
            held_object_ref="track_001",
        )
        held = self.verifier.verify_entry(
            instance, replace(current, physical=held_physical)
        )
        self.assertEqual(held.outcome, VerificationOutcome.PASSED)

    def test_surface_push_entry_reuses_public_visual_and_support_state(self) -> None:
        push_plan = make_task_plan_v1(
            task_id="push_to_region_001",
            instruction="把红色方块推入左区",
            planning_basis=self.plan.planning_basis,
            task=PushToRegionTaskV1(
                subject=self.plan.task.subject,
                destination=RegionRefV1("left_region", RegionType.REGION_2D),
            ),
        )
        push_program = compile_surface_push_capabilities_v1(
            push_plan, self.initial_scene
        )
        execution = start_capability_program_v1(
            self.program, initial_capability_execution_state_v1(self.program)
        )
        current = snapshot_for_execution(
            self.program,
            execution,
            observation_id=2,
            timestamp_s=0.6,
        )

        result = self.verifier.verify_entry(push_program.steps[0], current)

        self.assertEqual(result.outcome, VerificationOutcome.PASSED)
        self.assertEqual(
            result.contract_id, "subject_visible_supported_on_surface_v1"
        )

    def test_stale_visual_cannot_pass_visual_phase(self) -> None:
        execution = start_capability_program_v1(
            self.program, initial_capability_execution_state_v1(self.program)
        )
        execution = request_active_phase_verification_v1(self.program, execution)
        execution = apply_active_verification_v1(
            self.program, execution, VerificationOutcome.PASSED
        )
        execution = request_active_phase_verification_v1(self.program, execution)
        current = snapshot_for_execution(
            self.program,
            execution,
            observation_id=2,
            timestamp_s=0.6,
        )
        stale = replace(
            current,
            timestamp_s=1.0,
            sources=replace(
                current.sources,
                robot_measurement_timestamp_s=1.0,
            ),
            freshness=replace(
                current.freshness,
                visual_age_s=0.4,
                visual_is_fresh=False,
            ),
        )
        result = self.verifier.verify_active_phase(
            self.program, execution, stale
        )
        self.assertEqual(result.outcome, VerificationOutcome.PENDING)
        self.assertIn("fresh visual", result.reason)

    def test_placement_requires_three_distinct_stable_observations(self) -> None:
        program, execution = execution_at_place_verification()
        counts = []
        first_result = None
        for observation_id in (10, 10, 11, 12):
            current = snapshot_for_execution(
                program,
                execution,
                observation_id=observation_id,
                timestamp_s=(
                    1.0
                    + self.config.placement_minimum_observation_interval_s
                    * observation_id
                ),
                in_target=True,
            )
            current = replace(
                current,
                physical=replace(
                    current.physical,
                    grasp_state=GraspState.NOT_HELD,
                    held_object_ref=None,
                    evidence=PhysicalEvidenceV2(
                        robot_feedback_available=True,
                        visual_follow_confirmed=False,
                        fresh_visual_required=False,
                    ),
                ),
            )
            result = self.verifier.verify_active_phase(
                program, execution, current
            )
            if first_result is None:
                first_result = result
            counts.append(result.consecutive_pass_count)

        self.assertEqual(counts, [1, 1, 2, 3])
        self.assertEqual(first_result.outcome, VerificationOutcome.PENDING)
        self.assertEqual(result.outcome, VerificationOutcome.PASSED)
        self.assertEqual(result.required_consecutive_passes, 3)
        self.assertEqual(
            active_skill_state_v2(program, execution).status,
            SkillRunStatus.AWAITING_VERIFICATION,
        )


if __name__ == "__main__":
    unittest.main()
