"""将合法 TaskPlan 编译为通用能力程序；不执行动作。"""

from __future__ import annotations

from dataclasses import replace
from jev4mujoco.contracts.task_plan import (
    ArrangeObjectsTaskV1,
    ClearRegionTaskV1,
    PickAndPlaceTaskV1,
    PickObjectTaskV1,
    PlaceInsideTaskV1,
    PlaceRelativeTaskV1,
    PressButtonTaskV1,
    OrderedSequenceTaskV1,
    PushAsideTaskV1,
    PushToRegionTaskV1,
    SortObjectsTaskV1,
    StackObjectsTaskV1,
    TaskKind,
    TaskPlanV1,
    make_task_plan_v1,
    validate_task_plan_v1,
)
from jev4mujoco.contracts.visual_scene import VisualPredicate, VisualSceneV1

from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1,
    CapabilityKind,
    CapabilityGoalV1,
    CapabilityInstanceV1,
    CapabilityProgramV1,
)

def compile_pick_and_place_capabilities_v1(
    plan: TaskPlanV1,
    initial_scene: VisualSceneV1,
) -> CapabilityProgramV1:
    validate_task_plan_v1(plan, initial_scene)
    if plan.task.kind is not TaskKind.PICK_AND_PLACE or not isinstance(
        plan.task, PickAndPlaceTaskV1
    ):
        raise ValueError("V1 capability compiler currently supports pick_and_place only")
    subject_ref = plan.task.subject.initial_binding.track_ids[0]
    target_ref = plan.task.destination.region_ref
    entities = {entity.track_id: entity for entity in initial_scene.entities}
    steps = tuple(
        _region_placement_steps_v1(
            task_id=plan.task_id,
            subject_ref=subject_ref,
            target_ref=target_ref,
            source_ref=entities[subject_ref].support_region_ref,
            start_ordinal=0,
        )
    )
    return CapabilityProgramV1(
        program_id=f"{plan.task_id}:capabilities_v1",
        task_id=plan.task_id,
        planning_scene_id=plan.planning_basis.scene_id,
        steps=steps,
    )


def _region_placement_steps_v1(
    *,
    task_id: str,
    subject_ref: str,
    source_ref: str | None,
    target_ref: str,
    start_ordinal: int,
) -> list[CapabilityInstanceV1]:
    """Instantiate one reusable pick/transport/place chain for one subject."""
    if source_ref is None:
        raise ValueError("region placement subject must initially have a support region")
    steps: list[CapabilityInstanceV1] = []
    for offset, kind in enumerate(
        (CapabilityKind.PICK, CapabilityKind.TRANSPORT, CapabilityKind.PLACE)
    ):
        ordinal = start_ordinal + offset
        steps.append(
            CapabilityInstanceV1(
                instance_id=f"{task_id}:{ordinal}:{kind.value}",
                template_id=CAPABILITY_TEMPLATES_V1[kind].template_id,
                kind=kind,
                subject_ref=subject_ref,
                source_ref=source_ref,
                target_ref=None if kind is CapabilityKind.PICK else target_ref,
                ordinal=ordinal,
                goal=(
                    CapabilityGoalV1(
                        predicate=VisualPredicate.IN_REGION,
                        reference_ref=target_ref,
                        desired_value=True,
                    )
                    if kind in (CapabilityKind.TRANSPORT, CapabilityKind.PLACE)
                    else None
                ),
            )
        )
    return steps


def compile_repeated_region_placement_capabilities_v1(
    plan: TaskPlanV1,
    initial_scene: VisualSceneV1,
) -> CapabilityProgramV1:
    """Expand sorting/clearing into repeated generic region-placement chains."""
    validate_task_plan_v1(plan, initial_scene)
    assignments: list[tuple[str, str]] = []
    if isinstance(plan.task, SortObjectsTaskV1):
        for rule in plan.task.rules:
            assignments.extend(
                (subject_ref, rule.destination.region_ref)
                for subject_ref in rule.subjects.initial_binding.track_ids
            )
    elif isinstance(plan.task, ClearRegionTaskV1):
        assignments.extend(
            (subject_ref, plan.task.destination.region_ref)
            for subject_ref in plan.task.subjects.initial_binding.track_ids
        )
    elif isinstance(plan.task, ArrangeObjectsTaskV1):
        assignments.extend(
            (subject_ref, plan.task.workspace.region_ref)
            for subject_ref in plan.task.subjects.initial_binding.track_ids
        )
    else:
        raise ValueError(
            "repeated region placement compiler supports sort, clear, and arrange tasks"
        )

    entities = {entity.track_id: entity for entity in initial_scene.entities}
    steps: list[CapabilityInstanceV1] = []
    for subject_ref, target_ref in assignments:
        steps.extend(
            _region_placement_steps_v1(
                task_id=plan.task_id,
                subject_ref=subject_ref,
                source_ref=entities[subject_ref].support_region_ref,
                target_ref=target_ref,
                start_ordinal=len(steps),
            )
        )
    return CapabilityProgramV1(
        program_id=f"{plan.task_id}:capabilities_v1",
        task_id=plan.task_id,
        planning_scene_id=plan.planning_basis.scene_id,
        steps=tuple(steps),
    )


def compile_relative_placement_capabilities_v1(
    plan: TaskPlanV1,
    initial_scene: VisualSceneV1,
) -> CapabilityProgramV1:
    validate_task_plan_v1(plan, initial_scene)
    if not isinstance(plan.task, PlaceRelativeTaskV1):
        raise ValueError("relative placement compiler supports place_relative only")
    subject_ref = plan.task.subject.initial_binding.track_ids[0]
    reference_ref = plan.task.reference.initial_binding.track_ids[0]
    entities = {entity.track_id: entity for entity in initial_scene.entities}
    source_ref = entities[subject_ref].support_region_ref
    if source_ref is None:
        raise ValueError("relative placement subject must have an initial support")
    steps: list[CapabilityInstanceV1] = []
    for ordinal, kind in enumerate(
        (CapabilityKind.PICK, CapabilityKind.TRANSPORT, CapabilityKind.PLACE)
    ):
        goal = (
            CapabilityGoalV1(plan.task.relation, reference_ref, True)
            if kind in (CapabilityKind.TRANSPORT, CapabilityKind.PLACE)
            else None
        )
        steps.append(
            CapabilityInstanceV1(
                instance_id=f"{plan.task_id}:{ordinal}:{kind.value}",
                template_id=CAPABILITY_TEMPLATES_V1[kind].template_id,
                kind=kind,
                subject_ref=subject_ref,
                source_ref=source_ref,
                target_ref=None if kind is CapabilityKind.PICK else reference_ref,
                ordinal=ordinal,
                goal=goal,
            )
        )
    return CapabilityProgramV1(
        program_id=f"{plan.task_id}:capabilities_v1",
        task_id=plan.task_id,
        planning_scene_id=plan.planning_basis.scene_id,
        steps=tuple(steps),
    )


def compile_stack_capabilities_v1(
    plan: TaskPlanV1,
    initial_scene: VisualSceneV1,
) -> CapabilityProgramV1:
    validate_task_plan_v1(plan, initial_scene)
    if not isinstance(plan.task, StackObjectsTaskV1):
        raise ValueError("stack compiler supports stack_objects only")
    subject_refs = plan.task.subjects.initial_binding.track_ids
    entities = {entity.track_id: entity for entity in initial_scene.entities}
    steps: list[CapabilityInstanceV1] = []
    support_ref = subject_refs[0]
    for subject_ref in subject_refs[1:]:
        source_ref = entities[subject_ref].support_region_ref
        if source_ref is None:
            raise ValueError("stack subject must initially have a support")
        for kind in (
            CapabilityKind.PICK,
            CapabilityKind.TRANSPORT,
            CapabilityKind.PLACE,
        ):
            ordinal = len(steps)
            goal = (
                CapabilityGoalV1(VisualPredicate.ON_SURFACE, support_ref, True)
                if kind in (CapabilityKind.TRANSPORT, CapabilityKind.PLACE)
                else None
            )
            steps.append(
                CapabilityInstanceV1(
                    instance_id=f"{plan.task_id}:{ordinal}:{kind.value}",
                    template_id=CAPABILITY_TEMPLATES_V1[kind].template_id,
                    kind=kind,
                    subject_ref=subject_ref,
                    source_ref=source_ref,
                    target_ref=(
                        None if kind is CapabilityKind.PICK else support_ref
                    ),
                    ordinal=ordinal,
                    goal=goal,
                )
            )
        support_ref = subject_ref
    return CapabilityProgramV1(
        program_id=f"{plan.task_id}:capabilities_v1",
        task_id=plan.task_id,
        planning_scene_id=plan.planning_basis.scene_id,
        steps=tuple(steps),
    )


def compile_place_inside_capabilities_v1(
    plan: TaskPlanV1,
    initial_scene: VisualSceneV1,
) -> CapabilityProgramV1:
    validate_task_plan_v1(plan, initial_scene)
    if not isinstance(plan.task, PlaceInsideTaskV1):
        raise ValueError("place-inside compiler supports place_inside only")
    subject_ref = plan.task.subject.initial_binding.track_ids[0]
    source_ref = next(
        entity.support_region_ref
        for entity in initial_scene.entities
        if entity.track_id == subject_ref
    )
    if source_ref is None:
        raise ValueError("place-inside subject must initially have a support")
    target_ref = plan.task.interior_region.region_ref
    steps: list[CapabilityInstanceV1] = []
    for ordinal, kind in enumerate(
        (CapabilityKind.PICK, CapabilityKind.TRANSPORT, CapabilityKind.PLACE)
    ):
        goal = (
            CapabilityGoalV1(VisualPredicate.INSIDE, target_ref, True)
            if kind in (CapabilityKind.TRANSPORT, CapabilityKind.PLACE)
            else None
        )
        steps.append(
            CapabilityInstanceV1(
                instance_id=f"{plan.task_id}:{ordinal}:{kind.value}",
                template_id=CAPABILITY_TEMPLATES_V1[kind].template_id,
                kind=kind,
                subject_ref=subject_ref,
                source_ref=source_ref,
                target_ref=None if kind is CapabilityKind.PICK else target_ref,
                ordinal=ordinal,
                goal=goal,
            )
        )
    return CapabilityProgramV1(
        program_id=f"{plan.task_id}:capabilities_v1",
        task_id=plan.task_id,
        planning_scene_id=plan.planning_basis.scene_id,
        steps=tuple(steps),
    )


def compile_pick_capabilities_v1(
    plan: TaskPlanV1,
    initial_scene: VisualSceneV1,
) -> CapabilityProgramV1:
    validate_task_plan_v1(plan, initial_scene)
    if not isinstance(plan.task, PickObjectTaskV1):
        raise ValueError("pick compiler supports pick_object only")
    subject_ref = plan.task.subject.initial_binding.track_ids[0]
    initial_entity = next(
        entity for entity in initial_scene.entities if entity.track_id == subject_ref
    )
    source_ref = initial_entity.support_region_ref
    if source_ref is None:
        raise ValueError("pick subject must initially have a support region")
    template = CAPABILITY_TEMPLATES_V1[CapabilityKind.PICK]
    return CapabilityProgramV1(
        program_id=f"{plan.task_id}:capabilities_v1",
        task_id=plan.task_id,
        planning_scene_id=plan.planning_basis.scene_id,
        steps=(
            CapabilityInstanceV1(
                instance_id=f"{plan.task_id}:0:pick",
                template_id=template.template_id,
                kind=CapabilityKind.PICK,
                subject_ref=subject_ref,
                source_ref=source_ref,
                target_ref=None,
                ordinal=0,
            ),
        ),
    )


def compile_surface_push_capabilities_v1(
    plan: TaskPlanV1,
    initial_scene: VisualSceneV1,
) -> CapabilityProgramV1:
    """Compile different push tasks into the same parameterized capability."""
    validate_task_plan_v1(plan, initial_scene)
    if isinstance(plan.task, PushToRegionTaskV1):
        subject_refs = plan.task.subject.initial_binding.track_ids
        reference_ref = plan.task.destination.region_ref
        desired_value = True
    elif isinstance(plan.task, PushAsideTaskV1):
        subject_refs = plan.task.subjects.initial_binding.track_ids
        reference_ref = plan.task.protected_region.region_ref
        desired_value = True
    else:
        raise ValueError(
            "surface_push compiler supports push_to_region and push_aside only"
        )

    template = CAPABILITY_TEMPLATES_V1[CapabilityKind.SURFACE_PUSH]
    entities = {entity.track_id: entity for entity in initial_scene.entities}
    steps = []
    for ordinal, subject_ref in enumerate(subject_refs):
        source_ref = entities[subject_ref].support_region_ref
        if source_ref is None:
            raise ValueError("surface_push subject must initially have a support region")
        steps.append(
            CapabilityInstanceV1(
                instance_id=f"{plan.task_id}:{ordinal}:surface_push",
                template_id=template.template_id,
                kind=CapabilityKind.SURFACE_PUSH,
                subject_ref=subject_ref,
                source_ref=source_ref,
                target_ref=reference_ref,
                ordinal=ordinal,
                goal=CapabilityGoalV1(
                    predicate=(
                        VisualPredicate.IN_REGION
                        if isinstance(plan.task, PushToRegionTaskV1)
                        else VisualPredicate.CLEAR_OF_REGION
                    ),
                    reference_ref=reference_ref,
                    desired_value=desired_value,
                ),
            )
        )
    return CapabilityProgramV1(
        program_id=f"{plan.task_id}:capabilities_v1",
        task_id=plan.task_id,
        planning_scene_id=plan.planning_basis.scene_id,
        steps=tuple(steps),
    )


def compile_press_capabilities_v1(
    plan: TaskPlanV1,
    initial_scene: VisualSceneV1,
) -> CapabilityProgramV1:
    validate_task_plan_v1(plan, initial_scene)
    if not isinstance(plan.task, PressButtonTaskV1):
        raise ValueError("press compiler supports press_button only")
    subject_ref = plan.task.button.initial_binding.track_ids[0]
    target_ref = plan.task.press_point_region.region_ref
    template = CAPABILITY_TEMPLATES_V1[CapabilityKind.PRESS]
    return CapabilityProgramV1(
        program_id=f"{plan.task_id}:capabilities_v1",
        task_id=plan.task_id,
        planning_scene_id=plan.planning_basis.scene_id,
        steps=(
            CapabilityInstanceV1(
                instance_id=f"{plan.task_id}:0:press",
                template_id=template.template_id,
                kind=CapabilityKind.PRESS,
                subject_ref=subject_ref,
                source_ref=None,
                target_ref=target_ref,
                ordinal=0,
                goal=CapabilityGoalV1(
                    VisualPredicate.PRESSED,
                    target_ref,
                    True,
                ),
            ),
        ),
    )


def compile_task_capabilities_v1(
    plan: TaskPlanV1,
    initial_scene: VisualSceneV1,
) -> CapabilityProgramV1:
    """Dispatch a canonical TaskPlanV1 without embedding task-specific actions."""
    if isinstance(plan.task, OrderedSequenceTaskV1):
        validate_task_plan_v1(plan, initial_scene)
        steps: list[CapabilityInstanceV1] = []
        for stage in plan.task.stages:
            stage_plan = make_task_plan_v1(
                task_id=f"{plan.task_id}:{stage.stage_id}",
                instruction=plan.instruction,
                planning_basis=plan.planning_basis,
                task=stage.task,
            )
            stage_program = compile_task_capabilities_v1(stage_plan, initial_scene)
            for stage_step in stage_program.steps:
                ordinal = len(steps)
                steps.append(
                    replace(
                        stage_step,
                        instance_id=(
                            f"{plan.task_id}:{stage.stage_id}:"
                            f"{stage_step.ordinal}:{stage_step.kind.value}"
                        ),
                        ordinal=ordinal,
                    )
                )
        return CapabilityProgramV1(
            program_id=f"{plan.task_id}:capabilities_v1",
            task_id=plan.task_id,
            planning_scene_id=plan.planning_basis.scene_id,
            steps=tuple(steps),
        )
    if plan.task.kind is TaskKind.PICK_OBJECT:
        return compile_pick_capabilities_v1(plan, initial_scene)
    if plan.task.kind is TaskKind.PICK_AND_PLACE:
        return compile_pick_and_place_capabilities_v1(plan, initial_scene)
    if plan.task.kind in (
        TaskKind.SORT_OBJECTS,
        TaskKind.CLEAR_REGION,
        TaskKind.ARRANGE_OBJECTS,
    ):
        return compile_repeated_region_placement_capabilities_v1(plan, initial_scene)
    if plan.task.kind is TaskKind.PLACE_RELATIVE:
        return compile_relative_placement_capabilities_v1(plan, initial_scene)
    if plan.task.kind is TaskKind.STACK_OBJECTS:
        return compile_stack_capabilities_v1(plan, initial_scene)
    if plan.task.kind is TaskKind.PLACE_INSIDE:
        return compile_place_inside_capabilities_v1(plan, initial_scene)
    if plan.task.kind in (TaskKind.PUSH_TO_REGION, TaskKind.PUSH_ASIDE):
        return compile_surface_push_capabilities_v1(plan, initial_scene)
    if plan.task.kind is TaskKind.PRESS_BUTTON:
        return compile_press_capabilities_v1(plan, initial_scene)
    raise ValueError(
        f"TaskPlanV1 kind {plan.task.kind.value} has no V1 capability compiler"
    )
