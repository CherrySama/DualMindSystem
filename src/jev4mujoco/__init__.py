"""Capability-based task contracts and MuJoCo backend for JEV4Mujoco."""

from jev4mujoco.simulation.config import StageConfig, load_stage_config
from jev4mujoco.contracts.actions import (
    ActionRequest,
    ActionResult,
    ActionStatus,
    AtomicAction,
    PhysicalContactSnapshot,
    RawStateSnapshot,
)
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2
from jev4mujoco.contracts.task_plan import (
    TASK_REGISTRY_V1,
    TaskPlanV1,
    validate_task_plan_v1,
)
from jev4mujoco.contracts.visual_scene import VisualSceneV1
from jev4mujoco.perception.fixed_rgbd import (
    FixedRgbdCaptureV1,
    MujocoFixedRgbdConfigV1,
    MujocoFixedRgbdVisualSceneBuilderV1,
    load_mujoco_fixed_rgbd_config_v1,
)
from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1,
    CapabilityExecutionStateV1,
    CapabilityKind,
    CapabilityProgramV1,
)
from jev4mujoco.planning.compiler import (
    compile_pick_capabilities_v1,
    compile_pick_and_place_capabilities_v1,
    compile_place_inside_capabilities_v1,
    compile_press_capabilities_v1,
    compile_relative_placement_capabilities_v1,
    compile_repeated_region_placement_capabilities_v1,
    compile_stack_capabilities_v1,
    compile_surface_push_capabilities_v1,
    compile_task_capabilities_v1,
)
from jev4mujoco.runtime.capability_runtime import CapabilityRuntimeV1
from jev4mujoco.planning.presets import (
    build_preset_clear_region_plan_v1,
    build_preset_color_sort_plan_v1,
    build_preset_pick_and_place_plan_v1,
    build_preset_pick_object_plan_v1,
    build_preset_place_inside_plan_v1,
    build_preset_place_then_push_plan_v1,
    build_preset_press_button_plan_v1,
    build_preset_relative_placement_plan_v1,
    build_preset_row_arrangement_plan_v1,
    build_preset_stack_plan_v1,
    build_preset_push_aside_plan_v1,
    build_preset_push_to_region_plan_v1,
)
from jev4mujoco.runtime.verifier import RuntimeVerifierV1
from jev4mujoco.runtime.verifier_config import load_runtime_verifier_config_v1
from jev4mujoco.experiments.scenarios import (
    BlockPlaceInBowlRuntimeV1,
    BlockPickRuntimeV1,
    BlockSurfacePushRuntimeV1,
    LargeButtonPressRuntimeV1,
    MultiBlockArrangeRuntimeV1,
    MultiBlockClearRuntimeV1,
    MultiBlockSortRuntimeV1,
    MultiBlockStackRuntimeV1,
    OrderedPlaceThenPushRuntimeV1,
    RelativePlacementRuntimeV1,
)
from jev4mujoco.runtime.loop import BlockPickPlaceRuntimeV1
from jev4mujoco.experiments.results import BlockRuntimeOutcomeV1

__all__ = [
    "ActionRequest",
    "ActionResult",
    "ActionStatus",
    "AtomicAction",
    "CAPABILITY_TEMPLATES_V1",
    "CapabilityExecutionStateV1",
    "CapabilityKind",
    "CapabilityProgramV1",
    "CapabilityRuntimeV1",
    "BlockPickPlaceRuntimeV1",
    "BlockPlaceInBowlRuntimeV1",
    "BlockPickRuntimeV1",
    "BlockRuntimeOutcomeV1",
    "BlockSurfacePushRuntimeV1",
    "LargeButtonPressRuntimeV1",
    "MultiBlockArrangeRuntimeV1",
    "MultiBlockClearRuntimeV1",
    "MultiBlockSortRuntimeV1",
    "MultiBlockStackRuntimeV1",
    "OrderedPlaceThenPushRuntimeV1",
    "RelativePlacementRuntimeV1",
    "FixedRgbdCaptureV1",
    "PhysicalContactSnapshot",
    "RuntimeVerifierV1",
    "MujocoFixedRgbdConfigV1",
    "MujocoFixedRgbdVisualSceneBuilderV1",
    "RawStateSnapshot",
    "StageConfig",
    "StateSnapshotV2",
    "TASK_REGISTRY_V1",
    "TaskPlanV1",
    "VisualSceneV1",
    "build_preset_pick_and_place_plan_v1",
    "build_preset_clear_region_plan_v1",
    "build_preset_color_sort_plan_v1",
    "build_preset_pick_object_plan_v1",
    "build_preset_place_inside_plan_v1",
    "build_preset_place_then_push_plan_v1",
    "build_preset_press_button_plan_v1",
    "build_preset_relative_placement_plan_v1",
    "build_preset_row_arrangement_plan_v1",
    "build_preset_stack_plan_v1",
    "build_preset_push_aside_plan_v1",
    "build_preset_push_to_region_plan_v1",
    "compile_pick_capabilities_v1",
    "compile_pick_and_place_capabilities_v1",
    "compile_place_inside_capabilities_v1",
    "compile_press_capabilities_v1",
    "compile_relative_placement_capabilities_v1",
    "compile_repeated_region_placement_capabilities_v1",
    "compile_stack_capabilities_v1",
    "compile_surface_push_capabilities_v1",
    "compile_task_capabilities_v1",
    "load_stage_config",
    "load_mujoco_fixed_rgbd_config_v1",
    "load_runtime_verifier_config_v1",
    "validate_task_plan_v1",
]
