from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Any

from jev4mujoco.contracts.visual_scene import (
    EntityType,
    EntityV1,
    RegionType,
    Visibility,
    VisualPredicate,
    VisualSceneV1,
)


TASK_REGISTRY_VERSION = "task_registry_v1"


class TaskKind(str, Enum):
    PICK_OBJECT = "pick_object"
    PICK_AND_PLACE = "pick_and_place"
    SORT_OBJECTS = "sort_objects"
    CLEAR_REGION = "clear_region"
    PLACE_RELATIVE = "place_relative"
    PLACE_INSIDE = "place_inside"
    PUSH_TO_REGION = "push_to_region"
    PUSH_ASIDE = "push_aside"
    ARRANGE_OBJECTS = "arrange_objects"
    STACK_OBJECTS = "stack_objects"
    PRESS_BUTTON = "press_button"
    ORDERED_SEQUENCE = "ordered_sequence"


class Quantifier(str, Enum):
    ONE = "one"
    ALL_MATCHING = "all_matching"


class PopulationBasis(str, Enum):
    INITIAL_SCENE = "initial_scene"


class ArrangementPattern(str, Enum):
    ROW = "row"
    SEPARATED = "separated"
    GROUPED = "grouped"


class ArrangementOrdering(str, Enum):
    ANY = "any"


class ExpansionPolicy(str, Enum):
    SINGLE_SUBJECT = "single_subject"
    FOR_EACH_INITIAL_SUBJECT = "for_each_initial_subject"
    GOAL_LAYOUT_THEN_FOR_EACH = "goal_layout_then_for_each"
    ORDERED_STAGES = "ordered_stages"


class TaskImplementationStatus(str, Enum):
    SPECIFIED = "specified"
    IMPLEMENTED = "implemented"
    VALIDATED_IN_SIM = "validated_in_sim"
    VALIDATED_ON_REAL = "validated_on_real"


_RELATIVE_RELATIONS = frozenset(
    {
        VisualPredicate.LEFT_OF,
        VisualPredicate.RIGHT_OF,
        VisualPredicate.IN_FRONT_OF,
        VisualPredicate.NEAR,
        VisualPredicate.SEPARATED,
    }
)


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _to_jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_to_jsonable(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class SpatialScopeV1:
    relation: VisualPredicate
    reference: str

    def __post_init__(self) -> None:
        if self.relation not in (VisualPredicate.IN_REGION, VisualPredicate.ON_SURFACE):
            raise ValueError("selector spatial scope must be in_region or on_surface")
        _identifier(self.reference, "spatial_scope.reference")


@dataclass(frozen=True, slots=True)
class EntitySelectorV1:
    entity_type: EntityType
    category: str | None = None
    color: str | None = None
    spatial_scope: SpatialScopeV1 | None = None

    def __post_init__(self) -> None:
        if self.category is not None:
            _identifier(self.category, "selector.category")
        if self.color is not None:
            _identifier(self.color, "selector.color")


@dataclass(frozen=True, slots=True)
class InitialBindingV1:
    scene_id: str
    track_ids: tuple[str, ...]
    count: int

    def __post_init__(self) -> None:
        _identifier(self.scene_id, "initial_binding.scene_id")
        if type(self.count) is not int or self.count < 0:
            raise ValueError("initial_binding.count must be a nonnegative integer")
        if self.count != len(self.track_ids):
            raise ValueError("initial_binding.count must equal the number of track_ids")
        if len(self.track_ids) != len(set(self.track_ids)):
            raise ValueError("initial_binding.track_ids must be unique")
        for track_id in self.track_ids:
            _identifier(track_id, "initial_binding.track_id")


@dataclass(frozen=True, slots=True)
class EntitySetV1:
    selector: EntitySelectorV1
    quantifier: Quantifier
    initial_binding: InitialBindingV1

    def __post_init__(self) -> None:
        if self.quantifier is Quantifier.ONE and self.initial_binding.count != 1:
            raise ValueError("quantifier=one requires exactly one initial binding")


@dataclass(frozen=True, slots=True)
class RegionRefV1:
    region_ref: str
    expected_region_type: RegionType

    def __post_init__(self) -> None:
        _identifier(self.region_ref, "region_ref")


@dataclass(frozen=True, slots=True)
class PickObjectTaskV1:
    subject: EntitySetV1
    kind: TaskKind = field(default=TaskKind.PICK_OBJECT, init=False)


@dataclass(frozen=True, slots=True)
class PickAndPlaceTaskV1:
    subject: EntitySetV1
    destination: RegionRefV1
    kind: TaskKind = field(default=TaskKind.PICK_AND_PLACE, init=False)

    def __post_init__(self) -> None:
        if self.destination.expected_region_type is not RegionType.REGION_2D:
            raise ValueError("pick_and_place destination must be region_2d")


@dataclass(frozen=True, slots=True)
class SortRuleV1:
    rule_id: str
    subjects: EntitySetV1
    destination: RegionRefV1

    def __post_init__(self) -> None:
        _identifier(self.rule_id, "sort rule_id")
        if self.subjects.quantifier is not Quantifier.ALL_MATCHING:
            raise ValueError("sort rule subjects must use all_matching")
        if self.destination.expected_region_type is not RegionType.REGION_2D:
            raise ValueError("sort destination must be region_2d")


@dataclass(frozen=True, slots=True)
class SortObjectsTaskV1:
    rules: tuple[SortRuleV1, ...]
    population_basis: PopulationBasis = PopulationBasis.INITIAL_SCENE
    kind: TaskKind = field(default=TaskKind.SORT_OBJECTS, init=False)

    def __post_init__(self) -> None:
        if not self.rules:
            raise ValueError("sort_objects requires at least one rule")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("sort rule_ids must be unique")
        bound: set[str] = set()
        for rule in self.rules:
            overlap = bound & set(rule.subjects.initial_binding.track_ids)
            if overlap:
                raise ValueError(f"sort rules bind the same subjects: {sorted(overlap)}")
            bound.update(rule.subjects.initial_binding.track_ids)


@dataclass(frozen=True, slots=True)
class ClearRegionTaskV1:
    subjects: EntitySetV1
    source: RegionRefV1
    destination: RegionRefV1
    population_basis: PopulationBasis = PopulationBasis.INITIAL_SCENE
    kind: TaskKind = field(default=TaskKind.CLEAR_REGION, init=False)

    def __post_init__(self) -> None:
        if self.subjects.quantifier is not Quantifier.ALL_MATCHING:
            raise ValueError("clear_region subjects must use all_matching")
        if self.source.region_ref == self.destination.region_ref:
            raise ValueError("clear_region source and destination must differ")


@dataclass(frozen=True, slots=True)
class PlaceRelativeTaskV1:
    subject: EntitySetV1
    reference: EntitySetV1
    relation: VisualPredicate
    kind: TaskKind = field(default=TaskKind.PLACE_RELATIVE, init=False)

    def __post_init__(self) -> None:
        if self.subject.quantifier is not Quantifier.ONE or self.reference.quantifier is not Quantifier.ONE:
            raise ValueError("place_relative subject and reference must each be one entity")
        if self.relation not in _RELATIVE_RELATIONS:
            raise ValueError("place_relative relation is not supported")
        if self.subject.initial_binding.track_ids == self.reference.initial_binding.track_ids:
            raise ValueError("place_relative subject and reference must differ")


@dataclass(frozen=True, slots=True)
class PlaceInsideTaskV1:
    subject: EntitySetV1
    container: EntitySetV1
    interior_region: RegionRefV1
    kind: TaskKind = field(default=TaskKind.PLACE_INSIDE, init=False)

    def __post_init__(self) -> None:
        if self.subject.quantifier is not Quantifier.ONE:
            raise ValueError("place_inside subject must be one entity")
        if self.container.quantifier is not Quantifier.ONE:
            raise ValueError("place_inside container must be one entity")
        if self.container.selector.entity_type is not EntityType.CONTAINER:
            raise ValueError("place_inside container selector must target a container")
        if self.interior_region.expected_region_type is not RegionType.VOLUME_3D:
            raise ValueError("place_inside requires a volume_3d region")


@dataclass(frozen=True, slots=True)
class PushToRegionTaskV1:
    subject: EntitySetV1
    destination: RegionRefV1
    kind: TaskKind = field(default=TaskKind.PUSH_TO_REGION, init=False)

    def __post_init__(self) -> None:
        if self.subject.quantifier is not Quantifier.ONE:
            raise ValueError("push_to_region subject must be one entity")
        if self.destination.expected_region_type is not RegionType.REGION_2D:
            raise ValueError("push_to_region destination must be region_2d")


@dataclass(frozen=True, slots=True)
class PushAsideTaskV1:
    subjects: EntitySetV1
    protected_region: RegionRefV1
    population_basis: PopulationBasis = PopulationBasis.INITIAL_SCENE
    kind: TaskKind = field(default=TaskKind.PUSH_ASIDE, init=False)

    def __post_init__(self) -> None:
        if self.subjects.quantifier is not Quantifier.ALL_MATCHING:
            raise ValueError("push_aside subjects must use all_matching")
        if self.protected_region.expected_region_type is not RegionType.REGION_2D:
            raise ValueError("push_aside protected region must be region_2d")


@dataclass(frozen=True, slots=True)
class ArrangeObjectsTaskV1:
    subjects: EntitySetV1
    workspace: RegionRefV1
    pattern: ArrangementPattern
    ordering: ArrangementOrdering = ArrangementOrdering.ANY
    population_basis: PopulationBasis = PopulationBasis.INITIAL_SCENE
    kind: TaskKind = field(default=TaskKind.ARRANGE_OBJECTS, init=False)

    def __post_init__(self) -> None:
        if self.subjects.quantifier is not Quantifier.ALL_MATCHING:
            raise ValueError("arrange_objects subjects must use all_matching")
        if self.workspace.expected_region_type is not RegionType.REGION_2D:
            raise ValueError("arrangement workspace must be region_2d")


@dataclass(frozen=True, slots=True)
class StackObjectsTaskV1:
    subjects: EntitySetV1
    stacking_region: RegionRefV1
    population_basis: PopulationBasis = PopulationBasis.INITIAL_SCENE
    kind: TaskKind = field(default=TaskKind.STACK_OBJECTS, init=False)

    def __post_init__(self) -> None:
        if self.subjects.quantifier is not Quantifier.ALL_MATCHING:
            raise ValueError("stack_objects subjects must use all_matching")
        if self.subjects.initial_binding.count < 2:
            raise ValueError("stack_objects requires at least two initial subjects")
        if self.stacking_region.expected_region_type is not RegionType.SURFACE:
            raise ValueError("stacking_region must be a surface")


@dataclass(frozen=True, slots=True)
class PressButtonTaskV1:
    button: EntitySetV1
    press_point_region: RegionRefV1
    kind: TaskKind = field(default=TaskKind.PRESS_BUTTON, init=False)

    def __post_init__(self) -> None:
        if self.button.quantifier is not Quantifier.ONE:
            raise ValueError("press_button requires one button")
        if self.button.selector.entity_type is not EntityType.BUTTON:
            raise ValueError("press_button selector must target a button")
        if self.press_point_region.expected_region_type is not RegionType.POINT:
            raise ValueError("press_button requires a point region")


NonSequenceTaskSpecV1 = (
    PickObjectTaskV1
    | PickAndPlaceTaskV1
    | SortObjectsTaskV1
    | ClearRegionTaskV1
    | PlaceRelativeTaskV1
    | PlaceInsideTaskV1
    | PushToRegionTaskV1
    | PushAsideTaskV1
    | ArrangeObjectsTaskV1
    | StackObjectsTaskV1
    | PressButtonTaskV1
)


@dataclass(frozen=True, slots=True)
class OrderedStageV1:
    stage_id: str
    task: NonSequenceTaskSpecV1

    def __post_init__(self) -> None:
        _identifier(self.stage_id, "ordered stage_id")


@dataclass(frozen=True, slots=True)
class OrderedSequenceTaskV1:
    stages: tuple[OrderedStageV1, ...]
    kind: TaskKind = field(default=TaskKind.ORDERED_SEQUENCE, init=False)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("ordered_sequence requires at least one stage")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("ordered stage_ids must be unique")


TaskSpecV1 = NonSequenceTaskSpecV1 | OrderedSequenceTaskV1


@dataclass(frozen=True, slots=True)
class TaskDefinitionV1:
    task_kind: TaskKind
    expansion_policy: ExpansionPolicy
    required_skills: tuple[str, ...]
    completion_contract: str
    implementation_status: TaskImplementationStatus = TaskImplementationStatus.SPECIFIED

    def __post_init__(self) -> None:
        if not self.required_skills and self.task_kind is not TaskKind.ORDERED_SEQUENCE:
            raise ValueError("registered task must require at least one skill")
        for skill in self.required_skills:
            _identifier(skill, "required skill")
        _identifier(self.completion_contract, "completion_contract")


TASK_REGISTRY_V1: dict[TaskKind, TaskDefinitionV1] = {
    TaskKind.PICK_OBJECT: TaskDefinitionV1(
        TaskKind.PICK_OBJECT,
        ExpansionPolicy.SINGLE_SUBJECT,
        ("pick",),
        "pick_object_v1",
    ),
    TaskKind.PICK_AND_PLACE: TaskDefinitionV1(
        TaskKind.PICK_AND_PLACE,
        ExpansionPolicy.SINGLE_SUBJECT,
        ("pick", "transport", "place"),
        "pick_and_place_v1",
    ),
    TaskKind.SORT_OBJECTS: TaskDefinitionV1(
        TaskKind.SORT_OBJECTS,
        ExpansionPolicy.FOR_EACH_INITIAL_SUBJECT,
        ("pick", "transport", "place"),
        "sort_objects_v1",
    ),
    TaskKind.CLEAR_REGION: TaskDefinitionV1(
        TaskKind.CLEAR_REGION,
        ExpansionPolicy.FOR_EACH_INITIAL_SUBJECT,
        ("pick", "transport", "place"),
        "clear_region_v1",
    ),
    TaskKind.PLACE_RELATIVE: TaskDefinitionV1(
        TaskKind.PLACE_RELATIVE,
        ExpansionPolicy.SINGLE_SUBJECT,
        ("pick", "transport", "place"),
        "place_relative_v1",
    ),
    TaskKind.PLACE_INSIDE: TaskDefinitionV1(
        TaskKind.PLACE_INSIDE,
        ExpansionPolicy.SINGLE_SUBJECT,
        ("pick", "transport", "place"),
        "place_inside_v1",
    ),
    TaskKind.PUSH_TO_REGION: TaskDefinitionV1(
        TaskKind.PUSH_TO_REGION,
        ExpansionPolicy.SINGLE_SUBJECT,
        ("surface_push",),
        "push_to_region_v1",
    ),
    TaskKind.PUSH_ASIDE: TaskDefinitionV1(
        TaskKind.PUSH_ASIDE,
        ExpansionPolicy.FOR_EACH_INITIAL_SUBJECT,
        ("surface_push",),
        "push_aside_v1",
    ),
    TaskKind.ARRANGE_OBJECTS: TaskDefinitionV1(
        TaskKind.ARRANGE_OBJECTS,
        ExpansionPolicy.GOAL_LAYOUT_THEN_FOR_EACH,
        ("pick", "transport", "place"),
        "arrange_objects_v1",
    ),
    TaskKind.STACK_OBJECTS: TaskDefinitionV1(
        TaskKind.STACK_OBJECTS,
        ExpansionPolicy.GOAL_LAYOUT_THEN_FOR_EACH,
        ("pick", "transport", "place"),
        "stack_objects_v1",
    ),
    TaskKind.PRESS_BUTTON: TaskDefinitionV1(
        TaskKind.PRESS_BUTTON,
        ExpansionPolicy.SINGLE_SUBJECT,
        ("press_large_button",),
        "press_button_v1",
    ),
    TaskKind.ORDERED_SEQUENCE: TaskDefinitionV1(
        TaskKind.ORDERED_SEQUENCE,
        ExpansionPolicy.ORDERED_STAGES,
        (),
        "ordered_sequence_v1",
    ),
}


@dataclass(frozen=True, slots=True)
class PlanningBasisV1:
    scene_id: str
    observation_id: int
    capture_timestamp_s: float

    def __post_init__(self) -> None:
        _identifier(self.scene_id, "planning_basis.scene_id")
        if type(self.observation_id) is not int or self.observation_id <= 0:
            raise ValueError("planning_basis.observation_id must be a positive integer")
        if not isfinite(self.capture_timestamp_s) or self.capture_timestamp_s < 0.0:
            raise ValueError("planning_basis.capture_timestamp_s must be nonnegative")


@dataclass(frozen=True, slots=True)
class CompletionContractV1:
    contract_id: str
    authority: str = "runtime_verifier"

    def __post_init__(self) -> None:
        _identifier(self.contract_id, "completion.contract_id")
        if self.authority != "runtime_verifier":
            raise ValueError("completion authority must be runtime_verifier")


@dataclass(frozen=True, slots=True)
class TaskPlanV1:
    task_id: str
    instruction: str
    planning_basis: PlanningBasisV1
    task: TaskSpecV1
    completion: CompletionContractV1
    schema: str = "TaskPlanV1"
    schema_version: int = 1
    task_registry_version: str = TASK_REGISTRY_VERSION
    episode_mode: str = "single_task"

    def __post_init__(self) -> None:
        if self.schema != "TaskPlanV1" or self.schema_version != 1:
            raise ValueError("TaskPlanV1 schema identity is fixed")
        if self.task_registry_version != TASK_REGISTRY_VERSION:
            raise ValueError("unsupported task registry version")
        if self.episode_mode != "single_task":
            raise ValueError("TaskPlanV1 supports one task per episode")
        _identifier(self.task_id, "task_id")
        _identifier(self.instruction, "instruction")
        definition = TASK_REGISTRY_V1.get(self.task.kind)
        if definition is None:
            raise ValueError(f"unregistered task kind {self.task.kind}")
        if self.completion.contract_id != definition.completion_contract:
            raise ValueError(
                f"task {self.task.kind.value} requires completion contract "
                f"{definition.completion_contract}"
            )

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)


def make_task_plan_v1(
    *,
    task_id: str,
    instruction: str,
    planning_basis: PlanningBasisV1,
    task: TaskSpecV1,
) -> TaskPlanV1:
    definition = TASK_REGISTRY_V1[task.kind]
    return TaskPlanV1(
        task_id=task_id,
        instruction=instruction,
        planning_basis=planning_basis,
        task=task,
        completion=CompletionContractV1(definition.completion_contract),
    )


def _entity_sets(task: TaskSpecV1) -> tuple[EntitySetV1, ...]:
    if isinstance(task, PickObjectTaskV1):
        return (task.subject,)
    if isinstance(task, PickAndPlaceTaskV1):
        return (task.subject,)
    if isinstance(task, SortObjectsTaskV1):
        return tuple(rule.subjects for rule in task.rules)
    if isinstance(task, ClearRegionTaskV1):
        return (task.subjects,)
    if isinstance(task, PlaceRelativeTaskV1):
        return task.subject, task.reference
    if isinstance(task, PlaceInsideTaskV1):
        return task.subject, task.container
    if isinstance(task, PushToRegionTaskV1):
        return (task.subject,)
    if isinstance(task, PushAsideTaskV1):
        return (task.subjects,)
    if isinstance(task, ArrangeObjectsTaskV1):
        return (task.subjects,)
    if isinstance(task, StackObjectsTaskV1):
        return (task.subjects,)
    if isinstance(task, PressButtonTaskV1):
        return (task.button,)
    return tuple(
        entity_set
        for stage in task.stages
        for entity_set in _entity_sets(stage.task)
    )


def _region_refs(task: TaskSpecV1) -> tuple[RegionRefV1, ...]:
    if isinstance(task, PickObjectTaskV1):
        return ()
    if isinstance(task, PickAndPlaceTaskV1):
        return (task.destination,)
    if isinstance(task, SortObjectsTaskV1):
        return tuple(rule.destination for rule in task.rules)
    if isinstance(task, ClearRegionTaskV1):
        return task.source, task.destination
    if isinstance(task, PlaceRelativeTaskV1):
        return ()
    if isinstance(task, PlaceInsideTaskV1):
        return (task.interior_region,)
    if isinstance(task, PushToRegionTaskV1):
        return (task.destination,)
    if isinstance(task, PushAsideTaskV1):
        return (task.protected_region,)
    if isinstance(task, ArrangeObjectsTaskV1):
        return (task.workspace,)
    if isinstance(task, StackObjectsTaskV1):
        return (task.stacking_region,)
    if isinstance(task, PressButtonTaskV1):
        return (task.press_point_region,)
    return tuple(
        region_ref
        for stage in task.stages
        for region_ref in _region_refs(stage.task)
    )


def _matches_selector(
    entity: EntityV1,
    selector: EntitySelectorV1,
    scene: VisualSceneV1,
) -> bool:
    if entity.visibility is not Visibility.CLEAR or entity.entity_type is not selector.entity_type:
        return False
    if selector.category is not None and entity.semantic.category.value != selector.category:
        return False
    if selector.color is not None and entity.semantic.color.value != selector.color:
        return False
    if selector.spatial_scope is None:
        return True
    return any(
        relation.predicate is selector.spatial_scope.relation
        and relation.subject_ref == entity.track_id
        and relation.reference_ref == selector.spatial_scope.reference
        and relation.value
        for relation in scene.relations
    )


def validate_task_plan_v1(plan: TaskPlanV1, scene: VisualSceneV1) -> None:
    """Validate a canonical plan against the full initial VisualSceneV1."""
    if not scene.planning_ready:
        raise ValueError("initial VisualSceneV1 is not planning_ready")
    if plan.planning_basis.scene_id != scene.scene_id:
        raise ValueError("planning_basis.scene_id does not match VisualSceneV1")
    if plan.planning_basis.observation_id != scene.observation_id:
        raise ValueError("planning_basis.observation_id does not match VisualSceneV1")
    if plan.planning_basis.capture_timestamp_s != scene.capture_timestamp_s:
        raise ValueError("planning basis capture time does not match VisualSceneV1")

    entities = {entity.track_id: entity for entity in scene.entities}
    for entity_set in _entity_sets(plan.task):
        binding = entity_set.initial_binding
        if binding.scene_id != scene.scene_id:
            raise ValueError("initial binding was produced from a different scene")
        for track_id in binding.track_ids:
            entity = entities.get(track_id)
            if entity is None:
                raise ValueError(f"initial binding references unknown entity {track_id}")
            if not _matches_selector(entity, entity_set.selector, scene):
                raise ValueError(f"entity {track_id} does not match its selector")
        if entity_set.quantifier is Quantifier.ALL_MATCHING:
            expected = {
                entity.track_id
                for entity in scene.entities
                if _matches_selector(entity, entity_set.selector, scene)
            }
            actual = set(binding.track_ids)
            if actual != expected:
                raise ValueError(
                    "all_matching initial binding must equal the complete selector match set"
                )

    regions = {region.region_id: region for region in scene.regions}
    for reference in _region_refs(plan.task):
        region = regions.get(reference.region_ref)
        if region is None:
            raise ValueError(f"task references unknown region {reference.region_ref}")
        if region.region_type is not reference.expected_region_type:
            raise ValueError(
                f"region {reference.region_ref} has type {region.region_type.value}, "
                f"expected {reference.expected_region_type.value}"
            )

    if isinstance(plan.task, PlaceInsideTaskV1):
        container_id = plan.task.container.initial_binding.track_ids[0]
        interior = regions[plan.task.interior_region.region_ref]
        if interior.owner_entity_ref != container_id:
            raise ValueError("interior region is not owned by the selected container")
    if isinstance(plan.task, PressButtonTaskV1):
        button_id = plan.task.button.initial_binding.track_ids[0]
        point = regions[plan.task.press_point_region.region_ref]
        if point.owner_entity_ref != button_id:
            raise ValueError("press point is not owned by the selected button")
