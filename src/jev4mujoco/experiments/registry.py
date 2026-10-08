"""统一入口的实验装配信息；任务条件仍由场景类和 TaskPlan 定义。"""
from __future__ import annotations

from dataclasses import dataclass, field

from jev4mujoco.runtime.loop import BlockPickPlaceRuntimeV1
from jev4mujoco.experiments.scenarios import (
    BlockPickRuntimeV1,
    BlockPlaceInBowlRuntimeV1,
    BlockSurfacePushRuntimeV1,
    LargeButtonPressRuntimeV1,
    MultiBlockArrangeRuntimeV1,
    MultiBlockClearRuntimeV1,
    MultiBlockSortRuntimeV1,
    MultiBlockStackRuntimeV1,
    OrderedPlaceThenPushRuntimeV1,
    RelativePlacementRuntimeV1,
)


@dataclass(frozen=True, slots=True)
class ExperimentSpecV1:
    runtime_type: type[BlockPickPlaceRuntimeV1]
    run_name: str
    maximum_decisions: int
    result_task: str
    runtime_kwargs: dict[str, object] = field(default_factory=dict)


EXPERIMENTS_V1: dict[str, ExperimentSpecV1] = {
    'pick_and_place': ExperimentSpecV1(
        runtime_type=BlockPickPlaceRuntimeV1,
        run_name='v1_block_pick_and_place',
        maximum_decisions=140,
        result_task='pick_and_place_one_block',
    ),
    'pick': ExperimentSpecV1(
        runtime_type=BlockPickRuntimeV1,
        run_name='v1_block_pick',
        maximum_decisions=80,
        result_task='pick_one_block',
    ),
    'sort': ExperimentSpecV1(
        runtime_type=MultiBlockSortRuntimeV1,
        run_name='v1_two_block_sort',
        maximum_decisions=240,
        result_task='sort_two_blocks_by_color',
    ),
    'clear': ExperimentSpecV1(
        runtime_type=MultiBlockClearRuntimeV1,
        run_name='v1_two_block_clear',
        maximum_decisions=220,
        result_task='clear_two_blocks_into_storage_region',
    ),
    'relative': ExperimentSpecV1(
        runtime_type=RelativePlacementRuntimeV1,
        run_name='v1_block_relative_place',
        maximum_decisions=120,
        result_task='place_red_block_left_of_blue_block',
    ),
    'push_to_region': ExperimentSpecV1(
        runtime_type=BlockSurfacePushRuntimeV1,
        run_name='v1_block_surface_push',
        maximum_decisions=140,
        result_task='push_red_block_into_region',
        runtime_kwargs={'desired_inside_target': True},
    ),
    'push_aside': ExperimentSpecV1(
        runtime_type=BlockSurfacePushRuntimeV1,
        run_name='v1_block_surface_push_aside',
        maximum_decisions=140,
        result_task='push_red_block_clear_of_region',
        runtime_kwargs={'desired_inside_target': False},
    ),
    'arrange': ExperimentSpecV1(
        runtime_type=MultiBlockArrangeRuntimeV1,
        run_name='v1_three_block_arrange',
        maximum_decisions=320,
        result_task='arrange_three_blocks_in_a_row',
    ),
    'stack': ExperimentSpecV1(
        runtime_type=MultiBlockStackRuntimeV1,
        run_name='v1_two_block_stack',
        maximum_decisions=160,
        result_task='stack_red_block_on_blue_block',
    ),
    'bowl': ExperimentSpecV1(
        runtime_type=BlockPlaceInBowlRuntimeV1,
        run_name='v1_block_inside_bowl',
        maximum_decisions=180,
        result_task='place_red_block_inside_bowl',
    ),
    'press': ExperimentSpecV1(
        runtime_type=LargeButtonPressRuntimeV1,
        run_name='v1_large_button_press',
        maximum_decisions=100,
        result_task='press_large_yellow_button',
    ),
    'ordered': ExperimentSpecV1(
        runtime_type=OrderedPlaceThenPushRuntimeV1,
        run_name='v1_ordered_place_then_push',
        maximum_decisions=320,
        result_task='place_red_block_then_push_blue_block',
    ),
}
