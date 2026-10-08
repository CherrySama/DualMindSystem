from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from jev4mujoco.contracts.actions import PhysicalContactSnapshot, RawStateSnapshot
from jev4mujoco.contracts.state_snapshot import EffectorContactMode, GraspState
from jev4mujoco.contracts.visual_scene import (
    EvidenceStatus, RegionType, VisualPredicate, VisualRelationV1,
)
from jev4mujoco.experiments.scenarios import (
    MultiBlockClearRuntimeV1, MultiBlockSortRuntimeV1,
)
from jev4mujoco.planning.presets import (
    build_preset_clear_region_plan_v1, build_preset_color_sort_plan_v1,
    build_preset_push_to_region_plan_v1, build_preset_pick_object_plan_v1,
)
from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1, CapabilityKind, VerificationOutcome,
    CapabilityExecutionStateV1, CapabilityProgramStatus, CapabilityRunStatus, CapabilityStepStateV1,
)
from jev4mujoco.runtime.state_builder import LiveStateConfigV2, LiveStateSnapshotBuilderV2
from jev4mujoco.runtime.verifier import RuntimeVerifierV1
from jev4mujoco.runtime.verifier_config import load_runtime_verifier_config_v1
from tests.test_state_snapshot_v2 import snapshot
from tests.test_shared_subtask_context import phase_state, payload
from tests.test_visual_scene_v1 import measured


ROOT = Path(__file__).resolve().parents[1]


class FinalRegionGoalsTest(unittest.TestCase):
    def setUp(self):
        self.config = load_runtime_verifier_config_v1(ROOT / "configs/runtime_verifier_v1.yaml")
        self.initial = snapshot()
        first = self.initial.visual_scene.entities[0]
        second = replace(first, track_id="track_002",
            semantic=replace(first.semantic, color=measured("blue")))
        self.scene = replace(self.initial.visual_scene, entities=(first, second),
            capture_timestamp_s=0.5, publish_timestamp_s=0.5,
            regions=(*self.initial.visual_scene.regions,
                replace(self.initial.visual_scene.regions[0], region_id="work_region", region_type=RegionType.REGION_2D)),
            relations=tuple(VisualRelationV1(VisualPredicate.IN_REGION, ref, "work_region",
                True, EvidenceStatus.DERIVED, 1.0) for ref in ("track_001", "track_002")))

    def runtime(self, kind):
        runtime = object.__new__(kind)
        runtime._verifier = RuntimeVerifierV1(self.config)
        if kind is MultiBlockSortRuntimeV1:
            runtime.task_plan = build_preset_color_sort_plan_v1(
                task_id="round1_sort", instruction="分类", initial_scene=self.scene,
                color_destinations=(("red", "left_region"), ("blue", "left_region")))
        else:
            runtime.task_plan = build_preset_clear_region_plan_v1(
                task_id="round1_clear", instruction="清理", initial_scene=self.scene,
                source_ref="work_region", destination_ref="left_region")
        return runtime

    def final_state(self, first_inside=True, second_inside=True):
        relations = tuple(VisualRelationV1(VisualPredicate.IN_REGION, ref, "left_region",
            inside, EvidenceStatus.DERIVED, 1.0)
            for ref, inside in (("track_001", first_inside), ("track_002", second_inside)))
        entities = []
        for index, (entity, inside) in enumerate(zip(self.scene.entities, (first_inside, second_inside))):
            if inside:
                x, y = ((0.24, 0.09), (0.31, 0.16))[index]
                entity = replace(entity, geometry=replace(entity.geometry,
                    centroid_robot_base_m=measured((x, y, 0.03)),
                    top_center_robot_base_m=measured((x, y, 0.06)),
                    footprint_xy_robot_base_m=measured(((x-0.03, y-0.03),
                        (x+0.03, y-0.03), (x+0.03, y+0.03), (x-0.03, y+0.03)))))
            entities.append(entity)
        scene = replace(self.scene, relations=relations, entities=tuple(entities))
        return replace(self.initial, visual_scene=scene,
            sources=replace(self.initial.sources, visual_capture_timestamp_s=0.5),
            freshness=replace(self.initial.freshness, visual_age_s=0.0))

    def test_first_subject_moved_out_after_later_work_is_not_task_success(self):
        for kind in (MultiBlockSortRuntimeV1, MultiBlockClearRuntimeV1):
            with self.subTest(runtime=kind.__name__):
                passed, reason = self.runtime(kind)._runtime_task_completion(
                    self.final_state(first_inside=False))
                self.assertFalse(passed)
                self.assertIn("track_001", reason)

    def test_all_initially_bound_subjects_must_still_satisfy_their_goals(self):
        for kind in (MultiBlockSortRuntimeV1, MultiBlockClearRuntimeV1):
            with self.subTest(runtime=kind.__name__):
                self.assertTrue(self.runtime(kind)._runtime_task_completion(self.final_state())[0])
                self.assertFalse(self.runtime(kind)._runtime_task_completion(
                    self.final_state(second_inside=False))[0])

    def test_missing_subject_does_not_keep_its_old_success(self):
        state = self.final_state()
        state = replace(state, visual_scene=replace(state.visual_scene,
            entities=state.visual_scene.entities[:1],
            relations=tuple(r for r in state.visual_scene.relations if r.subject_ref != "track_002")))
        self.assertFalse(self.runtime(MultiBlockSortRuntimeV1)._runtime_task_completion(state)[0])

    def test_stale_observation_cannot_confirm_final_goals(self):
        state = replace(self.final_state(), timestamp_s=1.0,
            freshness=replace(self.initial.freshness, visual_age_s=0.5))
        self.assertFalse(self.runtime(MultiBlockClearRuntimeV1)._runtime_task_completion(state)[0])

    def test_completed_program_is_rejected_and_logged_by_the_live_loop(self):
        class NoDecisionExpected:
            def decide(self, state):
                raise AssertionError("a completed program must undergo final verification")

        with MultiBlockSortRuntimeV1(ROOT) as runtime:
            runtime.install_policy(NoDecisionExpected())
            runtime._capability_runtime._execution = CapabilityExecutionStateV1(
                runtime.program.program_id, CapabilityProgramStatus.COMPLETED, None,
                tuple(CapabilityStepStateV1(step.instance_id, CapabilityRunStatus.COMPLETED,
                    len(CAPABILITY_TEMPLATES_V1[step.kind].phases) - 1)
                    for step in runtime.program.steps), 0)
            # Original scene is outside the destinations; only the logical completion is injected.
            outcome = runtime.run(maximum_decisions=1, maximum_retries=0)
            self.assertFalse(outcome.runtime_verifier_success)
            checks = [event.payload for event in outcome.events
                if event.kind == "task_completion_verification"]
            self.assertTrue(checks)
            self.assertFalse(checks[0]["passed"])
            self.assertIn("block_1", checks[0]["reason"])
            self.assertEqual(checks[-1]["based_on_observation_id"], outcome.final_state.visual_scene.observation_id)


class ContactGraspConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.config = load_runtime_verifier_config_v1(ROOT / "configs/runtime_verifier_v1.yaml")

    def build(self, kind, *, right=True, opening=0.0008, target=0.0):
        phase = "establish_contact" if kind is CapabilityKind.SURFACE_PUSH else "close_gripper"
        state, program, execution = phase_state(kind, phase)
        scene = state.visual_scene
        if kind is CapabilityKind.SURFACE_PUSH:
            plan = build_preset_push_to_region_plan_v1(task_id=program.task_id,
                instruction="接触与抓持诊断", initial_scene=scene,
                subject_ref="track_001", destination_ref="left_region")
        else:
            plan = build_preset_pick_object_plan_v1(task_id=program.task_id,
                instruction="接触与抓持诊断", initial_scene=scene, subject_ref="track_001")
        raw = RawStateSnapshot(state_id=8, action_epoch=2, simulation_time_s=0.5,
            joint_positions_rad=state.robot.joint_positions_rad,
            joint_velocities_rad_s=state.robot.joint_velocities_rad_s,
            gripper_joint_m=opening, gripper_target_m=target,
            tcp_position_world_m=(0.27, -0.10, 0.02),
            tcp_position_robot_base_m=(0.27, -0.10, 0.02),
            tcp_orientation_robot_base_wxyz=state.robot.tcp_orientation_robot_base_wxyz,
            bowl_position_world_m=(0.30, -0.10, 0.03),
            bowl_position_robot_base_m=(0.30, -0.10, 0.03),
            bowl_orientation_robot_base_wxyz=(1.0, 0.0, 0.0, 0.0),
            bowl_linear_velocity_world_mps=(0.0, 0.0, 0.0),
            bowl_angular_velocity_world_radps=(0.0, 0.0, 0.0),
            target_position_robot_base_m=None,
            contacts=PhysicalContactSnapshot(True, right, 1.0, 1.0 if right else 0.0, True, 1.0))
        builder = LiveStateSnapshotBuilderV2(LiveStateConfigV2(0.04, 0.0, 0.002))
        current = builder.build(raw=raw, visual_scene=scene, task_plan=plan,
            program=program, execution=execution)
        return current, builder, raw, plan, program, execution

    def test_single_and_two_finger_push_contacts_do_not_imply_grasping(self):
        for right in (False, True):
            with self.subTest(right_contact=right):
                state, _, _, _, program, execution = self.build(CapabilityKind.SURFACE_PUSH, right=right)
                self.assertIs(state.physical.grasp_state, GraspState.NOT_HELD)
                self.assertIs(state.physical.effector_contact.mode, EffectorContactMode.PUSH_CONTACT)
                result = RuntimeVerifierV1(self.config).verify_active_phase(program, execution, state)
                self.assertIs(result.outcome, VerificationOutcome.PASSED)

    def test_blocked_closed_gripper_remains_uncertain_even_in_push_mode(self):
        state, _, _, _, program, execution = self.build(CapabilityKind.SURFACE_PUSH, opening=0.02)
        self.assertIs(state.physical.grasp_state, GraspState.CANDIDATE_HELD)
        self.assertIsNone(state.physical.held_object_ref)
        self.assertIs(state.physical.effector_contact.mode, EffectorContactMode.GRASP_CONTACT)
        self.assertIsNot(RuntimeVerifierV1(self.config).verify_active_phase(
            program, execution, state).outcome, VerificationOutcome.PASSED)

    def test_open_gripper_brushing_both_sides_is_not_a_grasp_candidate(self):
        state, _, _, _, program, execution = self.build(CapabilityKind.PICK, opening=0.04, target=0.04)
        self.assertIs(state.physical.grasp_state, GraspState.NOT_HELD)
        self.assertIsNot(RuntimeVerifierV1(self.config).verify_active_phase(
            program, execution, state).outcome, VerificationOutcome.PASSED)

    def test_real_grasp_evidence_still_needs_coupled_lift_and_stable_following(self):
        state, builder, raw, plan, program, execution = self.build(CapabilityKind.PICK, opening=0.02)
        self.assertIs(state.physical.grasp_state, GraspState.CANDIDATE_HELD)
        for timestamp in (0.6, 0.8):
            entity = state.visual_scene.entities[0]
            entity = replace(entity, geometry=replace(entity.geometry,
                centroid_robot_base_m=measured((0.30, -0.10, 0.04)),
                top_center_robot_base_m=measured((0.30, -0.10, 0.07))))
            scene = replace(state.visual_scene, observation_id=state.visual_scene.observation_id + 1,
                capture_timestamp_s=timestamp, publish_timestamp_s=timestamp, entities=(entity,))
            lifted = replace(raw, state_id=raw.state_id + 1, simulation_time_s=timestamp,
                tcp_position_robot_base_m=(0.27, -0.10, 0.03),
                bowl_position_robot_base_m=(0.30, -0.10, 0.04),
                contacts=replace(raw.contacts, bowl_table_contact=False))
            state = builder.build(raw=lifted, visual_scene=scene, task_plan=plan,
                program=program, execution=execution)
            self.assertIs(state.physical.grasp_state,
                GraspState.CANDIDATE_HELD if timestamp == 0.6 else GraspState.HELD)
        self.assertEqual(state.physical.held_object_ref, "track_001")
        self.assertTrue(state.physical.evidence.visual_follow_confirmed)


class PhaseContractConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.config = load_runtime_verifier_config_v1(ROOT / "configs/runtime_verifier_v1.yaml")

    def assess(self, state, program, execution):
        verifier = RuntimeVerifierV1(self.config)
        result = verifier.verify_active_phase(program, execution, state)
        feedback = verifier.decision_feedback(program.steps[0], state, result)
        context = payload(replace(state, phase_feedback=feedback))
        return result, context["phase"]

    def test_all_twenty_phase_condition_lists_agree_with_verification_in_fresh_states(self):
        count = 0
        for kind, template in CAPABILITY_TEMPLATES_V1.items():
            for phase in template.phases:
                with self.subTest(kind=kind.value, phase=phase.phase_id):
                    state, program, execution = phase_state(kind, phase.phase_id)
                    result, context = self.assess(state, program, execution)
                    self.assertEqual(result.outcome is VerificationOutcome.PASSED,
                        all(item["satisfied"] is True for item in context["conditions"]))
                    self.assertEqual(context["verification"]["outcome"], result.outcome.value)
                    count += 1
        self.assertEqual(count, 20)

    def test_press_completion_cannot_ignore_lost_point_alignment(self):
        state, program, execution = phase_state(CapabilityKind.PRESS, "press_button")
        point = replace(state.visual_scene.regions[1], region_type=RegionType.POINT,
            owner_entity_ref="track_001", geometry=replace(state.visual_scene.regions[1].geometry,
                boundary_xy_robot_base_m=None, surface_height_robot_base_m=None,
                point_robot_base_m=measured((0.42, -0.12, 0.055))))
        pressed = VisualRelationV1(VisualPredicate.PRESSED, "track_001", "left_region",
            True, EvidenceStatus.DERIVED, 1.0)
        state = replace(state, visual_scene=replace(state.visual_scene,
            regions=(state.visual_scene.regions[0], point), relations=(pressed,)),
            physical=replace(state.physical, effector_contact=replace(state.physical.effector_contact,
                mode=EffectorContactMode.PUSH_CONTACT, subject_ref="track_001")))
        for tcp, expected in (((0.42, -0.12, 0.02), True), ((0.45, -0.12, 0.02), False)):
            with self.subTest(tcp=tcp):
                current = replace(state, robot=replace(state.robot, tcp_position_robot_base_m=tcp))
                result, context = self.assess(current, program, execution)
                self.assertEqual(result.outcome is VerificationOutcome.PASSED, expected)
                alignment = next(item for item in context["conditions"]
                    if item["measurement"] == "tcp_press_point_xy_error_m")
                self.assertEqual(alignment["satisfied"], expected)

    def test_probe_lift_signal_cannot_override_lost_grasp_contact(self):
        state, program, execution = phase_state(CapabilityKind.PICK, "probe_grasp")
        state = replace(state, physical=replace(state.physical,
            grasp_state=GraspState.NOT_HELD,
            evidence=replace(state.physical.evidence, grasp_probe_lifted=True, grasp_probe_failed=False)))
        result, context = self.assess(state, program, execution)
        self.assertIs(result.outcome, VerificationOutcome.FAILED)
        self.assertFalse(all(item["satisfied"] is True for item in context["conditions"]))


if __name__ == "__main__":
    unittest.main()
