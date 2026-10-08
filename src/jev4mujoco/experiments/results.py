"""事件、回合结果及独立物理验收的数据类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from jev4mujoco.contracts.actions import RawStateSnapshot
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2

RuntimeEventSink = Callable[[object], None]


@dataclass(frozen=True, slots=True)
class BlockRuntimeEventV1:
    kind: str
    payload: object


@dataclass(frozen=True, slots=True)
class HiddenTruthResultV1:
    success: bool
    footprint_inside_target: bool
    supported: bool
    stable: bool
    fingers_clear: bool
    gripper_open: bool
    object_center_robot_base_m: tuple[float, float, float]
    object_linear_speed_mps: float
    reason: str


@dataclass(frozen=True, slots=True)
class SurfacePushHiddenTruthResultV1:
    success: bool
    footprint_inside_target: bool
    footprint_clear_of_target: bool
    target_clearance_m: float
    required_clearance_m: float
    supported: bool
    stable: bool
    effector_clear: bool
    object_center_robot_base_m: tuple[float, float, float]
    object_linear_speed_mps: float
    reason: str


@dataclass(frozen=True, slots=True)
class PickHiddenTruthResultV1:
    success: bool
    both_fingers_contact: bool
    gripper_blocked: bool
    gripper_commanded_closed: bool
    object_near_tcp: bool
    stable: bool
    object_center_robot_base_m: tuple[float, float, float]
    object_linear_speed_mps: float
    reason: str


@dataclass(frozen=True, slots=True)
class ObjectRegionTruthV1:
    subject_ref: str
    target_ref: str
    footprint_inside_target: bool
    supported: bool
    stable: bool
    fingers_clear: bool
    object_center_robot_base_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MultiObjectHiddenTruthResultV1:
    success: bool
    objects: tuple[ObjectRegionTruthV1, ...]
    gripper_open: bool
    reason: str


@dataclass(frozen=True, slots=True)
class RelativePlacementHiddenTruthResultV1:
    success: bool
    relation: str
    relation_satisfied: bool
    supported: bool
    stable: bool
    reference_supported: bool
    reference_stable: bool
    reference_displacement_m: float
    reference_displacement_within_limit: bool
    fingers_clear: bool
    gripper_open: bool
    subject_center_robot_base_m: tuple[float, float, float]
    reference_center_robot_base_m: tuple[float, float, float]
    reason: str


@dataclass(frozen=True, slots=True)
class ArrangementHiddenTruthResultV1:
    success: bool
    objects: tuple[ObjectRegionTruthV1, ...]
    row_cross_axis_span_m: float
    minimum_pairwise_clearance_m: float
    gripper_open: bool
    reason: str


@dataclass(frozen=True, slots=True)
class StackHiddenTruthResultV1:
    success: bool
    on_surface: bool
    support_overlap_fraction: float
    top_supported: bool
    top_stable: bool
    base_supported: bool
    base_stable: bool
    base_displacement_m: float
    base_displacement_within_limit: bool
    fingers_clear: bool
    gripper_open: bool
    top_center_robot_base_m: tuple[float, float, float]
    base_center_robot_base_m: tuple[float, float, float]
    reason: str


@dataclass(frozen=True, slots=True)
class ContainerPlacementHiddenTruthResultV1:
    success: bool
    fully_inside: bool
    supported_by_container_floor: bool
    stable: bool
    fingers_clear: bool
    gripper_open: bool
    object_center_robot_base_m: tuple[float, float, float]
    object_linear_speed_mps: float
    reason: str


@dataclass(frozen=True, slots=True)
class ClutterPickHiddenTruthResultV1:
    success: bool
    selected: PickHiddenTruthResultV1
    distractor_supported: bool
    distractor_stable: bool
    distractor_displacement_m: float
    distractor_displacement_within_limit: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ButtonPressHiddenTruthResultV1:
    success: bool
    press_travel_m: float
    travel_reached: bool
    effector_contact: bool
    button_supported: bool
    button_stable: bool
    gripper_closed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class OrderedSequenceHiddenTruthResultV1:
    success: bool
    objects: tuple[ObjectRegionTruthV1, ...]
    stage_order: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class BlockRuntimeOutcomeV1:
    completed: bool
    blocked: bool
    reason: str
    decision_count: int
    observation_count: int
    runtime_verifier_success: bool
    hidden_ground_truth: (
        HiddenTruthResultV1
        | SurfacePushHiddenTruthResultV1
        | PickHiddenTruthResultV1
        | MultiObjectHiddenTruthResultV1
        | RelativePlacementHiddenTruthResultV1
        | ArrangementHiddenTruthResultV1
        | StackHiddenTruthResultV1
        | ContainerPlacementHiddenTruthResultV1
        | ClutterPickHiddenTruthResultV1
        | ButtonPressHiddenTruthResultV1
        | OrderedSequenceHiddenTruthResultV1
    )
    overall_success: bool
    final_state: StateSnapshotV2
    final_raw_state: RawStateSnapshot
    events: tuple[BlockRuntimeEventV1, ...]
