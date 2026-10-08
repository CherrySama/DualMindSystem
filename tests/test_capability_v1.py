from __future__ import annotations

from dataclasses import replace
import unittest

from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1,
    CapabilityKind,
    CapabilityProgramStatus,
    CapabilityRunStatus,
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
    compile_pick_capabilities_v1,
    compile_place_inside_capabilities_v1,
    compile_relative_placement_capabilities_v1,
    compile_repeated_region_placement_capabilities_v1,
    compile_stack_capabilities_v1,
    compile_surface_push_capabilities_v1,
)
from jev4mujoco.planning.presets import build_preset_pick_object_plan_v1
from jev4mujoco.contracts.state_snapshot import SkillRunStatus
from jev4mujoco.contracts.task_plan import (
    ArrangementOrdering,
    ArrangementPattern,
    ArrangeObjectsTaskV1,
    ClearRegionTaskV1,
    EntitySelectorV1,
    EntitySetV1,
    InitialBindingV1,
    PickAndPlaceTaskV1,
    PlaceInsideTaskV1,
    PlaceRelativeTaskV1,
    PlanningBasisV1,
    Quantifier,
    RegionRefV1,
    SortObjectsTaskV1,
    SortRuleV1,
    StackObjectsTaskV1,
    SpatialScopeV1,
    PushAsideTaskV1,
    PushToRegionTaskV1,
    TASK_REGISTRY_V1,
    TaskKind,
    make_task_plan_v1,
)
from jev4mujoco.contracts.visual_scene import (
    EntityType,
    EvidenceStatus,
    EvidenceValueV1,
    RegionGeometryV1,
    RegionSourceKind,
    RegionSourceV1,
    RegionV1,
    RegionType,
    VisualPredicate,
    VisualRelationV1,
)
from tests.test_visual_scene_v1 import scene


def pick_and_place_plan():
    initial_scene = scene()
    subject = EntitySetV1(
        selector=EntitySelectorV1(
            entity_type=EntityType.MOVABLE_OBJECT,
            category="block",
            color="red",
        ),
        quantifier=Quantifier.ONE,
        initial_binding=InitialBindingV1(
            scene_id=initial_scene.scene_id,
            track_ids=("track_001",),
            count=1,
        ),
    )
    plan = make_task_plan_v1(
        task_id="pick_place_001",
        instruction="把红色方块放进左区",
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=PickAndPlaceTaskV1(
            subject=subject,
            destination=RegionRefV1("left_region", RegionType.REGION_2D),
        ),
    )
    return plan, initial_scene


def multi_object_scene():
    base = scene()
    red = base.entities[0]
    blue = replace(
        red,
        track_id="track_002",
        semantic=replace(red.semantic, color=replace(red.semantic.color, value="blue")),
    )
    left = base.regions[1]
    right = replace(
        left,
        region_id="right_region",
        semantic_label="right_sorting_region",
        geometry=replace(
            left.geometry,
            boundary_xy_robot_base_m=replace(
                left.geometry.boundary_xy_robot_base_m,
                value=((0.2, -0.20), (0.35, -0.20), (0.35, -0.05), (0.2, -0.05)),
            ),
        ),
    )
    source = replace(
        left,
        region_id="work_region",
        semantic_label="work_region",
        geometry=replace(
            left.geometry,
            boundary_xy_robot_base_m=replace(
                left.geometry.boundary_xy_robot_base_m,
                value=((0.4, -0.2), (0.6, -0.2), (0.6, 0.2), (0.4, 0.2)),
            ),
        ),
    )
    relations = list(base.relations)
    relations.extend(
        (
            VisualRelationV1(
                VisualPredicate.ON_SURFACE,
                "track_002",
                "table_surface",
                True,
                EvidenceStatus.DERIVED,
                0.99,
            ),
            VisualRelationV1(
                VisualPredicate.IN_REGION,
                "track_001",
                "work_region",
                True,
                EvidenceStatus.DERIVED,
                0.99,
            ),
            VisualRelationV1(
                VisualPredicate.IN_REGION,
                "track_002",
                "work_region",
                True,
                EvidenceStatus.DERIVED,
                0.99,
            ),
        )
    )
    return replace(
        base,
        entities=(red, blue),
        regions=(base.regions[0], left, right, source),
        relations=tuple(relations),
    )


class CapabilityV1Test(unittest.TestCase):
    def test_place_inside_reuses_pick_transport_place_with_inside_goal(self) -> None:
        initial_scene = multi_object_scene()
        container = replace(
            initial_scene.entities[1],
            track_id="container_1",
            entity_type=EntityType.CONTAINER,
            semantic=replace(
                initial_scene.entities[1].semantic,
                category=replace(
                    initial_scene.entities[1].semantic.category,
                    value="open_box",
                ),
            ),
        )
        measured = lambda value: EvidenceValueV1(
            value, EvidenceStatus.MEASURED, 1.0
        )
        interior = RegionV1(
            region_id="container_interior",
            region_type=RegionType.VOLUME_3D,
            semantic_label="open_box_interior",
            owner_entity_ref="container_1",
            geometry=RegionGeometryV1(
                boundary_xy_robot_base_m=measured(
                    ((0.25, 0.10), (0.40, 0.10), (0.40, 0.25), (0.25, 0.25))
                ),
                surface_height_robot_base_m=measured(0.01),
                surface_normal_robot_base=measured((0.0, 0.0, 1.0)),
                minimum_z_robot_base_m=measured(0.01),
                maximum_z_robot_base_m=measured(0.15),
            ),
            source=RegionSourceV1(RegionSourceKind.CALIBRATED, 1.0),
        )
        initial_scene = replace(
            initial_scene,
            entities=(initial_scene.entities[0], container),
            regions=(*initial_scene.regions, interior),
            relations=tuple(
                relation
                for relation in initial_scene.relations
                if "track_002" not in (relation.subject_ref, relation.reference_ref)
            ),
        )
        subject = EntitySetV1(
            selector=EntitySelectorV1(
                EntityType.MOVABLE_OBJECT, category="block", color="red"
            ),
            quantifier=Quantifier.ONE,
            initial_binding=InitialBindingV1(
                initial_scene.scene_id, ("track_001",), 1
            ),
        )
        container_set = EntitySetV1(
            selector=EntitySelectorV1(
                EntityType.CONTAINER, category="open_box", color="blue"
            ),
            quantifier=Quantifier.ONE,
            initial_binding=InitialBindingV1(
                initial_scene.scene_id, ("container_1",), 1
            ),
        )
        plan = make_task_plan_v1(
            task_id="inside_001",
            instruction="把红色方块放进蓝色盒子",
            planning_basis=PlanningBasisV1(
                initial_scene.scene_id,
                initial_scene.observation_id,
                initial_scene.capture_timestamp_s,
            ),
            task=PlaceInsideTaskV1(
                subject=subject,
                container=container_set,
                interior_region=RegionRefV1(
                    "container_interior", RegionType.VOLUME_3D
                ),
            ),
        )

        program = compile_place_inside_capabilities_v1(plan, initial_scene)

        self.assertEqual(
            tuple(step.kind for step in program.steps),
            (CapabilityKind.PICK, CapabilityKind.TRANSPORT, CapabilityKind.PLACE),
        )
        for step in program.steps[1:]:
            self.assertEqual(step.target_ref, "container_interior")
            self.assertEqual(step.goal.predicate, VisualPredicate.INSIDE)
            self.assertEqual(step.goal.reference_ref, "container_interior")

    def test_stack_reuses_pick_transport_place_with_on_surface_goal(self) -> None:
        initial_scene = multi_object_scene()
        plan = make_task_plan_v1(
            task_id="stack_001",
            instruction="把蓝色方块叠到红色方块上",
            planning_basis=PlanningBasisV1(
                initial_scene.scene_id,
                initial_scene.observation_id,
                initial_scene.capture_timestamp_s,
            ),
            task=StackObjectsTaskV1(
                subjects=EntitySetV1(
                    selector=EntitySelectorV1(EntityType.MOVABLE_OBJECT),
                    quantifier=Quantifier.ALL_MATCHING,
                    initial_binding=InitialBindingV1(
                        initial_scene.scene_id,
                        ("track_001", "track_002"),
                        2,
                    ),
                ),
                stacking_region=RegionRefV1("table_surface", RegionType.SURFACE),
            ),
        )

        program = compile_stack_capabilities_v1(plan, initial_scene)

        self.assertEqual(
            tuple(step.kind for step in program.steps),
            (CapabilityKind.PICK, CapabilityKind.TRANSPORT, CapabilityKind.PLACE),
        )
        self.assertEqual(tuple(step.subject_ref for step in program.steps), ("track_002",) * 3)
        self.assertIsNone(program.steps[0].goal)
        for step in program.steps[1:]:
            self.assertEqual(step.target_ref, "track_001")
            self.assertEqual(step.goal.predicate, VisualPredicate.ON_SURFACE)
            self.assertEqual(step.goal.reference_ref, "track_001")
            self.assertTrue(step.goal.desired_value)

    def test_arrangement_reuses_repeated_region_placement_chains(self) -> None:
        initial_scene = multi_object_scene()
        plan = make_task_plan_v1(
            task_id="arrange_001",
            instruction="把方块排成一行",
            planning_basis=PlanningBasisV1(
                initial_scene.scene_id,
                initial_scene.observation_id,
                initial_scene.capture_timestamp_s,
            ),
            task=ArrangeObjectsTaskV1(
                subjects=EntitySetV1(
                    selector=EntitySelectorV1(EntityType.MOVABLE_OBJECT),
                    quantifier=Quantifier.ALL_MATCHING,
                    initial_binding=InitialBindingV1(
                        initial_scene.scene_id,
                        ("track_001", "track_002"),
                        2,
                    ),
                ),
                workspace=RegionRefV1("work_region", RegionType.REGION_2D),
                pattern=ArrangementPattern.ROW,
                ordering=ArrangementOrdering.ANY,
            ),
        )

        program = compile_repeated_region_placement_capabilities_v1(
            plan, initial_scene
        )

        self.assertEqual(len(program.steps), 6)
        self.assertEqual(
            tuple(step.kind for step in program.steps),
            (CapabilityKind.PICK, CapabilityKind.TRANSPORT, CapabilityKind.PLACE) * 2,
        )

    def test_relative_placement_parameterizes_transport_and_place_goals(self) -> None:
        initial_scene = multi_object_scene()
        subject = EntitySetV1(
            selector=EntitySelectorV1(
                EntityType.MOVABLE_OBJECT, category="block", color="red"
            ),
            quantifier=Quantifier.ONE,
            initial_binding=InitialBindingV1(
                initial_scene.scene_id, ("track_001",), 1
            ),
        )
        reference = EntitySetV1(
            selector=EntitySelectorV1(
                EntityType.MOVABLE_OBJECT, category="block", color="blue"
            ),
            quantifier=Quantifier.ONE,
            initial_binding=InitialBindingV1(
                initial_scene.scene_id, ("track_002",), 1
            ),
        )
        plan = make_task_plan_v1(
            task_id="relative_001",
            instruction="把红块放到蓝块左边",
            planning_basis=PlanningBasisV1(
                initial_scene.scene_id,
                initial_scene.observation_id,
                initial_scene.capture_timestamp_s,
            ),
            task=PlaceRelativeTaskV1(
                subject=subject,
                reference=reference,
                relation=VisualPredicate.LEFT_OF,
            ),
        )

        program = compile_relative_placement_capabilities_v1(plan, initial_scene)

        self.assertEqual(
            tuple(step.kind for step in program.steps),
            (CapabilityKind.PICK, CapabilityKind.TRANSPORT, CapabilityKind.PLACE),
        )
        self.assertIsNone(program.steps[0].goal)
        for step in program.steps[1:]:
            self.assertEqual(step.target_ref, "track_002")
            self.assertEqual(step.goal.predicate, VisualPredicate.LEFT_OF)
            self.assertEqual(step.goal.reference_ref, "track_002")
            self.assertTrue(step.goal.desired_value)

    def test_sort_expands_rules_into_reused_region_placement_chains(self) -> None:
        initial_scene = multi_object_scene()
        basis = PlanningBasisV1(
            initial_scene.scene_id,
            initial_scene.observation_id,
            initial_scene.capture_timestamp_s,
        )
        rules = []
        for rule_id, track_id, color, destination in (
            ("red_to_left", "track_001", "red", "left_region"),
            ("blue_to_right", "track_002", "blue", "right_region"),
        ):
            rules.append(
                SortRuleV1(
                    rule_id=rule_id,
                    subjects=EntitySetV1(
                        selector=EntitySelectorV1(
                            EntityType.MOVABLE_OBJECT,
                            category="block",
                            color=color,
                        ),
                        quantifier=Quantifier.ALL_MATCHING,
                        initial_binding=InitialBindingV1(
                            initial_scene.scene_id,
                            (track_id,),
                            1,
                        ),
                    ),
                    destination=RegionRefV1(destination, RegionType.REGION_2D),
                )
            )
        plan = make_task_plan_v1(
            task_id="sort_001",
            instruction="红色放左区，蓝色放右区",
            planning_basis=basis,
            task=SortObjectsTaskV1(tuple(rules)),
        )

        program = compile_repeated_region_placement_capabilities_v1(
            plan, initial_scene
        )

        self.assertEqual(len(program.steps), 6)
        self.assertEqual(
            tuple(step.kind for step in program.steps),
            (
                CapabilityKind.PICK,
                CapabilityKind.TRANSPORT,
                CapabilityKind.PLACE,
                CapabilityKind.PICK,
                CapabilityKind.TRANSPORT,
                CapabilityKind.PLACE,
            ),
        )
        self.assertEqual(
            tuple(step.subject_ref for step in program.steps),
            ("track_001",) * 3 + ("track_002",) * 3,
        )
        self.assertEqual(program.steps[2].goal.reference_ref, "left_region")
        self.assertEqual(program.steps[5].goal.reference_ref, "right_region")

    def test_clear_region_reuses_same_chain_for_each_initial_subject(self) -> None:
        initial_scene = multi_object_scene()
        plan = make_task_plan_v1(
            task_id="clear_001",
            instruction="清空工作区",
            planning_basis=PlanningBasisV1(
                initial_scene.scene_id,
                initial_scene.observation_id,
                initial_scene.capture_timestamp_s,
            ),
            task=ClearRegionTaskV1(
                subjects=EntitySetV1(
                    selector=EntitySelectorV1(
                        EntityType.MOVABLE_OBJECT,
                        category="block",
                        spatial_scope=SpatialScopeV1(
                            VisualPredicate.IN_REGION,
                            "work_region",
                        ),
                    ),
                    quantifier=Quantifier.ALL_MATCHING,
                    initial_binding=InitialBindingV1(
                        initial_scene.scene_id,
                        ("track_001", "track_002"),
                        2,
                    ),
                ),
                source=RegionRefV1("work_region", RegionType.REGION_2D),
                destination=RegionRefV1("left_region", RegionType.REGION_2D),
            ),
        )

        program = compile_repeated_region_placement_capabilities_v1(
            plan, initial_scene
        )

        self.assertEqual(len(program.steps), 6)
        self.assertTrue(
            all(
                step.target_ref == "left_region"
                for step in program.steps
                if step.kind is not CapabilityKind.PICK
            )
        )
        self.assertTrue(
            all(
                step.goal is None
                or step.goal.predicate is VisualPredicate.IN_REGION
                for step in program.steps
            )
        )

    def test_pick_object_compiles_to_one_generic_pick_step(self) -> None:
        initial_scene = scene()
        plan = build_preset_pick_object_plan_v1(
            task_id="pick_001",
            instruction="抓取红色方块",
            initial_scene=initial_scene,
            subject_ref="track_001",
        )

        program = compile_pick_capabilities_v1(plan, initial_scene)

        self.assertEqual(len(program.steps), 1)
        self.assertEqual(program.steps[0].kind, CapabilityKind.PICK)
        self.assertEqual(program.steps[0].subject_ref, "track_001")
        self.assertEqual(program.steps[0].source_ref, "table_surface")
        self.assertIsNone(program.steps[0].target_ref)
        self.assertIsNone(program.steps[0].goal)

    def test_push_tasks_share_one_parameterized_surface_push_capability(self) -> None:
        initial_scene = scene()
        basis = PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        )
        single_subject = EntitySetV1(
            selector=EntitySelectorV1(
                entity_type=EntityType.MOVABLE_OBJECT,
                category="block",
                color="red",
            ),
            quantifier=Quantifier.ONE,
            initial_binding=InitialBindingV1(
                scene_id=initial_scene.scene_id,
                track_ids=("track_001",),
                count=1,
            ),
        )
        all_subjects = EntitySetV1(
            selector=single_subject.selector,
            quantifier=Quantifier.ALL_MATCHING,
            initial_binding=single_subject.initial_binding,
        )
        to_region = make_task_plan_v1(
            task_id="push_to_region_001",
            instruction="把红色方块推入左区",
            planning_basis=basis,
            task=PushToRegionTaskV1(
                subject=single_subject,
                destination=RegionRefV1("left_region", RegionType.REGION_2D),
            ),
        )
        aside = make_task_plan_v1(
            task_id="push_aside_001",
            instruction="把左区里的红色方块推开",
            planning_basis=basis,
            task=PushAsideTaskV1(
                subjects=all_subjects,
                protected_region=RegionRefV1("left_region", RegionType.REGION_2D),
            ),
        )

        into_program = compile_surface_push_capabilities_v1(to_region, initial_scene)
        outside_program = compile_surface_push_capabilities_v1(aside, initial_scene)
        into = into_program.steps[0]
        outside = outside_program.steps[0]

        self.assertEqual(
            TASK_REGISTRY_V1[TaskKind.PUSH_TO_REGION].required_skills,
            ("surface_push",),
        )
        self.assertEqual(
            TASK_REGISTRY_V1[TaskKind.PUSH_ASIDE].required_skills,
            ("surface_push",),
        )
        self.assertEqual(into.kind, CapabilityKind.SURFACE_PUSH)
        self.assertEqual(outside.kind, CapabilityKind.SURFACE_PUSH)
        self.assertTrue(
            CAPABILITY_TEMPLATES_V1[CapabilityKind.SURFACE_PUSH].runtime_implemented
        )
        self.assertEqual(into.goal.predicate, VisualPredicate.IN_REGION)
        self.assertTrue(into.goal.desired_value)
        self.assertEqual(outside.goal.predicate, VisualPredicate.CLEAR_OF_REGION)
        self.assertTrue(outside.goal.desired_value)
        self.assertEqual(into.source_ref, "table_surface")
        self.assertEqual(into.target_ref, "left_region")
        self.assertFalse(hasattr(into, "push_distance_m"))
        self.assertFalse(hasattr(into, "push_direction"))

        active = start_capability_program_v1(
            into_program, initial_capability_execution_state_v1(into_program)
        )
        skill = active_skill_state_v2(into_program, active)
        self.assertEqual(skill.goal.predicate, VisualPredicate.IN_REGION)
        self.assertEqual(skill.goal.reference_ref, "left_region")
        self.assertTrue(skill.goal.desired_value)

    def test_pick_and_place_explicitly_requires_three_capabilities(self) -> None:
        self.assertEqual(
            TASK_REGISTRY_V1[TaskKind.PICK_AND_PLACE].required_skills,
            ("pick", "transport", "place"),
        )
        plan, initial_scene = pick_and_place_plan()
        program = compile_pick_and_place_capabilities_v1(plan, initial_scene)

        self.assertEqual(
            tuple(step.kind for step in program.steps),
            (
                CapabilityKind.PICK,
                CapabilityKind.TRANSPORT,
                CapabilityKind.PLACE,
            ),
        )
        self.assertEqual(program.steps[0].subject_ref, "track_001")
        self.assertEqual(program.steps[0].source_ref, "table_surface")
        self.assertEqual(program.steps[1].source_ref, "table_surface")
        self.assertIsNone(program.steps[0].target_ref)
        self.assertEqual(program.steps[1].target_ref, "left_region")
        self.assertEqual(program.steps[2].target_ref, "left_region")

    def test_templates_expose_actions_but_verification_phases_do_not(self) -> None:
        self.assertEqual(set(CAPABILITY_TEMPLATES_V1), set(CapabilityKind))
        for template in CAPABILITY_TEMPLATES_V1.values():
            self.assertEqual(
                template.phases[-1].completion_contract,
                template.completion_contract,
            )
            for phase in template.phases:
                if phase.motor_action_allowed:
                    self.assertTrue(phase.allowed_actions)
                else:
                    self.assertEqual(phase.allowed_actions, ())
                    self.assertTrue(phase.requires_fresh_visual)

    def test_every_phase_requires_runtime_verification_before_advancing(self) -> None:
        plan, initial_scene = pick_and_place_plan()
        program = compile_pick_and_place_capabilities_v1(plan, initial_scene)
        state = start_capability_program_v1(
            program, initial_capability_execution_state_v1(program)
        )

        first_skill = active_skill_state_v2(program, state)
        self.assertEqual(first_skill.skill_kind, "pick")
        self.assertEqual(first_skill.phase, "prepare_gripper")
        self.assertEqual(first_skill.status, SkillRunStatus.RUNNING)
        self.assertIn("gripper_open", first_skill.allowed_actions)
        with self.assertRaisesRegex(ValueError, "not awaiting verification"):
            apply_active_verification_v1(
                program, state, VerificationOutcome.PASSED
            )

        state = request_active_phase_verification_v1(program, state)
        awaiting = active_skill_state_v2(program, state)
        self.assertEqual(awaiting.status, SkillRunStatus.AWAITING_VERIFICATION)
        self.assertEqual(awaiting.allowed_actions, ())
        pending = apply_active_verification_v1(
            program, state, VerificationOutcome.PENDING
        )
        self.assertIs(pending, state)

        state = apply_active_verification_v1(
            program, state, VerificationOutcome.PASSED
        )
        self.assertEqual(active_skill_state_v2(program, state).phase, "align_subject_xy")

        while state.status is not CapabilityProgramStatus.COMPLETED:
            if state.active_step_index is None:
                state = activate_next_capability_v1(program, state)
            state = request_active_phase_verification_v1(program, state)
            state = apply_active_verification_v1(
                program, state, VerificationOutcome.PASSED
            )

        self.assertIsNone(state.active_step_index)
        self.assertTrue(
            all(step.status is CapabilityRunStatus.COMPLETED for step in state.steps)
        )

    def test_next_capability_waits_for_explicit_entry_activation(self) -> None:
        plan, initial_scene = pick_and_place_plan()
        program = compile_pick_and_place_capabilities_v1(plan, initial_scene)
        state = start_capability_program_v1(
            program, initial_capability_execution_state_v1(program)
        )
        pick_phase_count = len(CAPABILITY_TEMPLATES_V1[CapabilityKind.PICK].phases)
        for _ in range(pick_phase_count):
            state = request_active_phase_verification_v1(program, state)
            state = apply_active_verification_v1(
                program, state, VerificationOutcome.PASSED
            )

        self.assertEqual(state.status, CapabilityProgramStatus.RUNNING)
        self.assertIsNone(state.active_step_index)
        self.assertEqual(state.steps[0].status, CapabilityRunStatus.COMPLETED)
        self.assertEqual(state.steps[1].status, CapabilityRunStatus.PENDING)

        state = activate_next_capability_v1(program, state)
        self.assertEqual(state.active_step_index, 1)
        self.assertEqual(
            active_skill_state_v2(program, state).skill_kind,
            "transport",
        )

    def test_failed_verification_blocks_without_starting_next_capability(self) -> None:
        plan, initial_scene = pick_and_place_plan()
        program = compile_pick_and_place_capabilities_v1(plan, initial_scene)
        state = start_capability_program_v1(
            program, initial_capability_execution_state_v1(program)
        )
        state = request_active_phase_verification_v1(program, state)
        state = apply_active_verification_v1(
            program,
            state,
            VerificationOutcome.FAILED,
            failure_reason="gripper did not open",
        )

        self.assertEqual(state.status, CapabilityProgramStatus.BLOCKED)
        self.assertEqual(state.active_step_index, 0)
        self.assertEqual(state.steps[0].status, CapabilityRunStatus.BLOCKED)
        self.assertEqual(state.steps[1].status, CapabilityRunStatus.PENDING)
        self.assertEqual(state.blocked_reason, "gripper did not open")


if __name__ == "__main__":
    unittest.main()
