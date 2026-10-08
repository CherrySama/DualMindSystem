from __future__ import annotations

from jev4mujoco.contracts.task_plan import (
    ArrangementOrdering,
    ArrangementPattern,
    ArrangeObjectsTaskV1,
    ClearRegionTaskV1,
    EntitySelectorV1,
    EntitySetV1,
    InitialBindingV1,
    OrderedSequenceTaskV1,
    OrderedStageV1,
    PickAndPlaceTaskV1,
    PickObjectTaskV1,
    PlaceInsideTaskV1,
    PressButtonTaskV1,
    PlaceRelativeTaskV1,
    PlanningBasisV1,
    PushAsideTaskV1,
    PushToRegionTaskV1,
    Quantifier,
    RegionRefV1,
    SortObjectsTaskV1,
    SortRuleV1,
    SpatialScopeV1,
    StackObjectsTaskV1,
    TaskPlanV1,
    make_task_plan_v1,
    validate_task_plan_v1,
)
from jev4mujoco.contracts.visual_scene import (
    EntityType,
    RegionType,
    Visibility,
    VisualPredicate,
    VisualSceneV1,
)


def build_preset_pick_object_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    subject_ref: str,
) -> TaskPlanV1:
    entity = next(
        (item for item in initial_scene.entities if item.track_id == subject_ref),
        None,
    )
    if entity is None or entity.visibility is not Visibility.CLEAR:
        raise ValueError("pick subject must be visible in the initial scene")
    category = entity.semantic.category.value
    color = entity.semantic.color.value
    if category is None or color is None:
        raise ValueError("preset pick requires category and color semantics")
    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=PickObjectTaskV1(
            subject=EntitySetV1(
                selector=EntitySelectorV1(
                    entity_type=entity.entity_type,
                    category=category,
                    color=color,
                ),
                quantifier=Quantifier.ONE,
                initial_binding=InitialBindingV1(
                    scene_id=initial_scene.scene_id,
                    track_ids=(subject_ref,),
                    count=1,
                ),
            )
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_pick_and_place_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    subject_ref: str,
    destination_ref: str,
) -> TaskPlanV1:
    """Build the canonical plan after a preset task selected scene references.

    This is the deterministic first-version planning adapter, not a language
    parser. A future slow brain may choose the same references, but it must
    still produce a plan accepted by validate_task_plan_v1.
    """

    entity = next(
        (
            candidate
            for candidate in initial_scene.entities
            if candidate.track_id == subject_ref
        ),
        None,
    )
    if entity is None:
        raise ValueError(f"unknown pick_and_place subject {subject_ref}")
    if entity.visibility is not Visibility.CLEAR:
        raise ValueError("pick_and_place subject must be visible in the initial scene")
    category = entity.semantic.category.value
    color = entity.semantic.color.value
    if category is None or color is None:
        raise ValueError("preset pick_and_place requires category and color semantics")

    destination = next(
        (
            region
            for region in initial_scene.regions
            if region.region_id == destination_ref
        ),
        None,
    )
    if destination is None:
        raise ValueError(f"unknown pick_and_place destination {destination_ref}")
    if destination.region_type is not RegionType.REGION_2D:
        raise ValueError("preset pick_and_place destination must be region_2d")

    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=PickAndPlaceTaskV1(
            subject=EntitySetV1(
                selector=EntitySelectorV1(
                    entity_type=entity.entity_type,
                    category=category,
                    color=color,
                ),
                quantifier=Quantifier.ONE,
                initial_binding=InitialBindingV1(
                    scene_id=initial_scene.scene_id,
                    track_ids=(subject_ref,),
                    count=1,
                ),
            ),
            destination=RegionRefV1(
                region_ref=destination_ref,
                expected_region_type=RegionType.REGION_2D,
            ),
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_color_sort_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    color_destinations: tuple[tuple[str, str], ...],
) -> TaskPlanV1:
    """Deterministic slow-brain substitute for a color-to-region sort instruction."""
    if not color_destinations:
        raise ValueError("color sort requires at least one color/destination mapping")
    if len({color for color, _ in color_destinations}) != len(color_destinations):
        raise ValueError("color sort mappings must use unique colors")
    regions = {region.region_id: region for region in initial_scene.regions}
    rules: list[SortRuleV1] = []
    for color, destination_ref in color_destinations:
        destination = regions.get(destination_ref)
        if destination is None or destination.region_type is not RegionType.REGION_2D:
            raise ValueError("color sort destination must be a region_2d")
        subjects = tuple(
            entity
            for entity in initial_scene.entities
            if entity.visibility is Visibility.CLEAR
            and entity.semantic.color.value == color
        )
        if not subjects:
            raise ValueError(f"color sort found no visible {color} subjects")
        categories = {entity.semantic.category.value for entity in subjects}
        entity_types = {entity.entity_type for entity in subjects}
        if len(categories) != 1 or None in categories or len(entity_types) != 1:
            raise ValueError("each color rule must resolve to one category and entity type")
        rules.append(
            SortRuleV1(
                rule_id=f"{color}_to_{destination_ref}",
                subjects=EntitySetV1(
                    selector=EntitySelectorV1(
                        entity_type=subjects[0].entity_type,
                        category=subjects[0].semantic.category.value,
                        color=color,
                    ),
                    quantifier=Quantifier.ALL_MATCHING,
                    initial_binding=InitialBindingV1(
                        scene_id=initial_scene.scene_id,
                        track_ids=tuple(entity.track_id for entity in subjects),
                        count=len(subjects),
                    ),
                ),
                destination=RegionRefV1(destination_ref, RegionType.REGION_2D),
            )
        )
    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=SortObjectsTaskV1(tuple(rules)),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_clear_region_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    source_ref: str,
    destination_ref: str,
) -> TaskPlanV1:
    """Bind every initially visible movable object in a source region."""
    regions = {region.region_id: region for region in initial_scene.regions}
    for reference in (source_ref, destination_ref):
        region = regions.get(reference)
        if region is None or region.region_type is not RegionType.REGION_2D:
            raise ValueError("clear source and destination must be region_2d")
    subject_refs = tuple(
        entity.track_id
        for entity in initial_scene.entities
        if entity.entity_type is EntityType.MOVABLE_OBJECT
        and entity.visibility is Visibility.CLEAR
        and any(
            relation.predicate is VisualPredicate.IN_REGION
            and relation.subject_ref == entity.track_id
            and relation.reference_ref == source_ref
            and relation.value
            for relation in initial_scene.relations
        )
    )
    if not subject_refs:
        raise ValueError("clear source contains no visible movable objects")
    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=ClearRegionTaskV1(
            subjects=EntitySetV1(
                selector=EntitySelectorV1(
                    entity_type=EntityType.MOVABLE_OBJECT,
                    spatial_scope=SpatialScopeV1(
                        relation=VisualPredicate.IN_REGION,
                        reference=source_ref,
                    ),
                ),
                quantifier=Quantifier.ALL_MATCHING,
                initial_binding=InitialBindingV1(
                    scene_id=initial_scene.scene_id,
                    track_ids=subject_refs,
                    count=len(subject_refs),
                ),
            ),
            source=RegionRefV1(source_ref, RegionType.REGION_2D),
            destination=RegionRefV1(destination_ref, RegionType.REGION_2D),
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_relative_placement_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    subject_ref: str,
    reference_ref: str,
    relation: VisualPredicate,
) -> TaskPlanV1:
    """Bind one subject/reference pair for a relation-conditioned placement."""
    entities = {entity.track_id: entity for entity in initial_scene.entities}
    subject = entities.get(subject_ref)
    reference = entities.get(reference_ref)
    if subject is None or reference is None or subject_ref == reference_ref:
        raise ValueError("relative placement requires two distinct visible entities")
    if (
        subject.visibility is not Visibility.CLEAR
        or reference.visibility is not Visibility.CLEAR
    ):
        raise ValueError("relative placement entities must be visible")

    def one(entity):
        return EntitySetV1(
            selector=EntitySelectorV1(
                entity_type=entity.entity_type,
                category=entity.semantic.category.value,
                color=entity.semantic.color.value,
            ),
            quantifier=Quantifier.ONE,
            initial_binding=InitialBindingV1(
                scene_id=initial_scene.scene_id,
                track_ids=(entity.track_id,),
                count=1,
            ),
        )

    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=PlaceRelativeTaskV1(
            subject=one(subject),
            reference=one(reference),
            relation=relation,
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_row_arrangement_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    subject_refs: tuple[str, ...],
    workspace_ref: str,
) -> TaskPlanV1:
    """Bind selected objects to a coarse row-layout objective in one workspace."""
    if len(subject_refs) < 2 or len(subject_refs) != len(set(subject_refs)):
        raise ValueError("row arrangement requires at least two unique subjects")
    entities = {entity.track_id: entity for entity in initial_scene.entities}
    subjects = tuple(entities.get(reference) for reference in subject_refs)
    if any(
        entity is None
        or entity.entity_type is not EntityType.MOVABLE_OBJECT
        or entity.visibility is not Visibility.CLEAR
        for entity in subjects
    ):
        raise ValueError("row arrangement subjects must be visible movable objects")
    workspace = next(
        (
            region
            for region in initial_scene.regions
            if region.region_id == workspace_ref
        ),
        None,
    )
    if workspace is None or workspace.region_type is not RegionType.REGION_2D:
        raise ValueError("row arrangement workspace must be region_2d")
    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=ArrangeObjectsTaskV1(
            subjects=EntitySetV1(
                selector=EntitySelectorV1(entity_type=EntityType.MOVABLE_OBJECT),
                quantifier=Quantifier.ALL_MATCHING,
                initial_binding=InitialBindingV1(
                    scene_id=initial_scene.scene_id,
                    track_ids=subject_refs,
                    count=len(subject_refs),
                ),
            ),
            workspace=RegionRefV1(workspace_ref, RegionType.REGION_2D),
            pattern=ArrangementPattern.ROW,
            ordering=ArrangementOrdering.ANY,
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_stack_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    subject_refs: tuple[str, ...],
    stacking_region_ref: str,
) -> TaskPlanV1:
    """Bind an ordered bottom-to-top set of blocks for a stack objective."""
    if len(subject_refs) < 2 or len(subject_refs) != len(set(subject_refs)):
        raise ValueError("stacking requires at least two unique subjects")
    entities = {entity.track_id: entity for entity in initial_scene.entities}
    subjects = tuple(entities.get(reference) for reference in subject_refs)
    if any(
        entity is None
        or entity.entity_type is not EntityType.MOVABLE_OBJECT
        or entity.visibility is not Visibility.CLEAR
        for entity in subjects
    ):
        raise ValueError("stacking subjects must be visible movable objects")
    stacking_region = next(
        (
            region
            for region in initial_scene.regions
            if region.region_id == stacking_region_ref
        ),
        None,
    )
    if stacking_region is None or stacking_region.region_type is not RegionType.SURFACE:
        raise ValueError("stacking region must be a surface")
    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=StackObjectsTaskV1(
            subjects=EntitySetV1(
                selector=EntitySelectorV1(entity_type=EntityType.MOVABLE_OBJECT),
                quantifier=Quantifier.ALL_MATCHING,
                initial_binding=InitialBindingV1(
                    scene_id=initial_scene.scene_id,
                    track_ids=subject_refs,
                    count=len(subject_refs),
                ),
            ),
            stacking_region=RegionRefV1(
                stacking_region_ref,
                RegionType.SURFACE,
            ),
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_place_inside_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    subject_ref: str,
    container_ref: str,
    interior_region_ref: str,
) -> TaskPlanV1:
    """Bind one movable subject and one visible container-owned interior volume."""
    entities = {entity.track_id: entity for entity in initial_scene.entities}
    subject = entities.get(subject_ref)
    container = entities.get(container_ref)
    if (
        subject is None
        or subject.entity_type is not EntityType.MOVABLE_OBJECT
        or subject.visibility is not Visibility.CLEAR
    ):
        raise ValueError("place-inside subject must be a visible movable object")
    if (
        container is None
        or container.entity_type is not EntityType.CONTAINER
        or container.visibility is not Visibility.CLEAR
    ):
        raise ValueError("place-inside container must be visible")
    interior = next(
        (
            region
            for region in initial_scene.regions
            if region.region_id == interior_region_ref
        ),
        None,
    )
    if (
        interior is None
        or interior.region_type is not RegionType.VOLUME_3D
        or interior.owner_entity_ref != container_ref
    ):
        raise ValueError("place-inside interior must be a container-owned volume")

    def one(entity):
        return EntitySetV1(
            selector=EntitySelectorV1(
                entity_type=entity.entity_type,
                category=entity.semantic.category.value,
                color=entity.semantic.color.value,
            ),
            quantifier=Quantifier.ONE,
            initial_binding=InitialBindingV1(
                scene_id=initial_scene.scene_id,
                track_ids=(entity.track_id,),
                count=1,
            ),
        )

    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=PlaceInsideTaskV1(
            subject=one(subject),
            container=one(container),
            interior_region=RegionRefV1(
                interior_region_ref,
                RegionType.VOLUME_3D,
            ),
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_press_button_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    button_ref: str,
    press_point_ref: str,
) -> TaskPlanV1:
    button = next(
        (entity for entity in initial_scene.entities if entity.track_id == button_ref),
        None,
    )
    if (
        button is None
        or button.entity_type is not EntityType.BUTTON
        or button.visibility is not Visibility.CLEAR
    ):
        raise ValueError("press target must be a visible button")
    point = next(
        (region for region in initial_scene.regions if region.region_id == press_point_ref),
        None,
    )
    if (
        point is None
        or point.region_type is not RegionType.POINT
        or point.owner_entity_ref != button_ref
    ):
        raise ValueError("press point must be owned by the selected button")
    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=PressButtonTaskV1(
            button=EntitySetV1(
                selector=EntitySelectorV1(
                    entity_type=EntityType.BUTTON,
                    category=button.semantic.category.value,
                    color=button.semantic.color.value,
                ),
                quantifier=Quantifier.ONE,
                initial_binding=InitialBindingV1(
                    scene_id=initial_scene.scene_id,
                    track_ids=(button_ref,),
                    count=1,
                ),
            ),
            press_point_region=RegionRefV1(press_point_ref, RegionType.POINT),
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_place_then_push_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    place_subject_ref: str,
    place_destination_ref: str,
    push_subject_ref: str,
    push_destination_ref: str,
) -> TaskPlanV1:
    """Compose two ordinary task specs into one ordered TaskPlanV1."""
    place_plan = build_preset_pick_and_place_plan_v1(
        task_id=f"{task_id}:place_stage",
        instruction=instruction,
        initial_scene=initial_scene,
        subject_ref=place_subject_ref,
        destination_ref=place_destination_ref,
    )
    push_plan = build_preset_push_to_region_plan_v1(
        task_id=f"{task_id}:push_stage",
        instruction=instruction,
        initial_scene=initial_scene,
        subject_ref=push_subject_ref,
        destination_ref=push_destination_ref,
    )
    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=OrderedSequenceTaskV1(
            stages=(
                OrderedStageV1("place_stage", place_plan.task),
                OrderedStageV1("push_stage", push_plan.task),
            )
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_push_to_region_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    subject_ref: str,
    destination_ref: str,
) -> TaskPlanV1:
    """Deterministic planning adapter for a goal-conditioned surface push."""
    entity = next(
        (item for item in initial_scene.entities if item.track_id == subject_ref),
        None,
    )
    if entity is None or entity.visibility is not Visibility.CLEAR:
        raise ValueError("surface push subject must be visible in the initial scene")
    category = entity.semantic.category.value
    color = entity.semantic.color.value
    if category is None or color is None:
        raise ValueError("preset surface push requires category and color semantics")
    destination = next(
        (region for region in initial_scene.regions if region.region_id == destination_ref),
        None,
    )
    if destination is None or destination.region_type is not RegionType.REGION_2D:
        raise ValueError("surface push destination must be a region_2d")
    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=PushToRegionTaskV1(
            subject=EntitySetV1(
                selector=EntitySelectorV1(
                    entity_type=entity.entity_type,
                    category=category,
                    color=color,
                ),
                quantifier=Quantifier.ONE,
                initial_binding=InitialBindingV1(
                    scene_id=initial_scene.scene_id,
                    track_ids=(subject_ref,),
                    count=1,
                ),
            ),
            destination=RegionRefV1(
                region_ref=destination_ref,
                expected_region_type=RegionType.REGION_2D,
            ),
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan


def build_preset_push_aside_plan_v1(
    *,
    task_id: str,
    instruction: str,
    initial_scene: VisualSceneV1,
    subject_ref: str,
    protected_region_ref: str,
) -> TaskPlanV1:
    """Build the inverse goal while preserving the same surface-push ability."""
    entity = next(
        (item for item in initial_scene.entities if item.track_id == subject_ref),
        None,
    )
    if entity is None or entity.visibility is not Visibility.CLEAR:
        raise ValueError("surface push subject must be visible in the initial scene")
    category = entity.semantic.category.value
    color = entity.semantic.color.value
    if category is None or color is None:
        raise ValueError("preset surface push requires category and color semantics")
    protected_region = next(
        (
            region
            for region in initial_scene.regions
            if region.region_id == protected_region_ref
        ),
        None,
    )
    if (
        protected_region is None
        or protected_region.region_type is not RegionType.REGION_2D
    ):
        raise ValueError("surface push protected region must be a region_2d")
    plan = make_task_plan_v1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=PlanningBasisV1(
            scene_id=initial_scene.scene_id,
            observation_id=initial_scene.observation_id,
            capture_timestamp_s=initial_scene.capture_timestamp_s,
        ),
        task=PushAsideTaskV1(
            subjects=EntitySetV1(
                selector=EntitySelectorV1(
                    entity_type=entity.entity_type,
                    category=category,
                    color=color,
                ),
                quantifier=Quantifier.ALL_MATCHING,
                initial_binding=InitialBindingV1(
                    scene_id=initial_scene.scene_id,
                    track_ids=(subject_ref,),
                    count=1,
                ),
            ),
            protected_region=RegionRefV1(
                region_ref=protected_region_ref,
                expected_region_type=RegionType.REGION_2D,
            ),
        ),
    )
    validate_task_plan_v1(plan, initial_scene)
    return plan
