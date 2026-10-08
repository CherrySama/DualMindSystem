"""现有实验场景装配与公开任务完成条件，共用同一个动作闭环。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from jev4mujoco.planning.compiler import (
    compile_surface_push_capabilities_v1,
)
from jev4mujoco.contracts.actions import RawStateSnapshot
from jev4mujoco.planning.presets import (
    build_preset_clear_region_plan_v1,
    build_preset_color_sort_plan_v1,
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
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2
from jev4mujoco.perception.surface_geometry import polygon_clearance_v1
from jev4mujoco.contracts.task_plan import TaskPlanV1
from jev4mujoco.contracts.visual_scene import VisualPredicate, VisualSceneV1

from jev4mujoco.experiments.results import (
    SurfacePushHiddenTruthResultV1,
    PickHiddenTruthResultV1,
    MultiObjectHiddenTruthResultV1,
    RelativePlacementHiddenTruthResultV1,
    ArrangementHiddenTruthResultV1,
    StackHiddenTruthResultV1,
    ContainerPlacementHiddenTruthResultV1,
    ButtonPressHiddenTruthResultV1,
    OrderedSequenceHiddenTruthResultV1,
    RuntimeEventSink,
)
from jev4mujoco.experiments.evaluation import (
    evaluate_pick,
    evaluate_surface_push,
    evaluate_sort,
    evaluate_clear,
    evaluate_relative_placement,
    evaluate_arrangement,
    evaluate_stack,
    evaluate_inside_bowl,
    evaluate_button_press,
    evaluate_ordered_sequence,
)
from jev4mujoco.runtime.loop import BlockPickPlaceRuntimeV1

class BlockPickRuntimeV1(BlockPickPlaceRuntimeV1):
    """Single generic pick capability on the same RGB-D/runtime loop."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
    ) -> None:
        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            return build_preset_pick_object_plan_v1(
                task_id="block_pick_001",
                instruction="抓取红色方块",
                initial_scene=scene,
                subject_ref="block_1",
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            visual_config_filename="pick_object.yaml",
            task_plan_builder=build_plan,
            episode_scene_id="block_pick_episode_001",
        )

    def _hidden_truth(self, raw: RawStateSnapshot) -> PickHiddenTruthResultV1:
        return evaluate_pick(self, raw)



class BlockSurfacePushRuntimeV1(BlockPickPlaceRuntimeV1):
    """Same live runtime loop with a different TaskPlan and completion oracle."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
        desired_inside_target: bool = True,
        object_initial_position_world_m: tuple[float, float, float] | None = None,
    ) -> None:
        self._desired_inside_target = desired_inside_target

        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            if desired_inside_target:
                return build_preset_push_to_region_plan_v1(
                    task_id="block_surface_push_into_001",
                    instruction="把红色方块推入目标区域",
                    initial_scene=scene,
                    subject_ref="block_1",
                    destination_ref="target_region",
                )
            return build_preset_push_aside_plan_v1(
                task_id="block_surface_push_aside_001",
                instruction="把红色方块推出目标区域",
                initial_scene=scene,
                subject_ref="block_1",
                protected_region_ref="target_region",
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            object_initial_position_world_m=object_initial_position_world_m,
            visual_config_filename=(
                "push_to_region.yaml" if desired_inside_target else "push_aside.yaml"
            ),
            task_plan_builder=build_plan,
            episode_scene_id="block_surface_push_episode_001",
        )
        self._install_task_program(
            self.task_plan,
            compile_surface_push_capabilities_v1(self.task_plan, self._last_visual_scene),
            gripper_position_tolerance_m=(
                self._verifier_config.surface_push_gripper_position_tolerance_m
            ),
        )

    def _hidden_truth(
        self, raw: RawStateSnapshot
    ) -> SurfacePushHiddenTruthResultV1:
        return evaluate_surface_push(self, raw)


class MultiBlockSortRuntimeV1(BlockPickPlaceRuntimeV1):
    """Ideal two-object color sort using the same generic capability runtime."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
    ) -> None:
        self._sort_destinations = {
            "block_1": "red_region",
            "block_2": "blue_region",
        }

        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            return build_preset_color_sort_plan_v1(
                task_id="two_block_color_sort_001",
                instruction="把红色方块放进红区，把蓝色方块放进蓝区",
                initial_scene=scene,
                color_destinations=(
                    ("red", "red_region"),
                    ("blue", "blue_region"),
                ),
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            visual_config_filename="sort_objects.yaml",
            object_body_names=("block_1", "block_2"),
            task_plan_builder=build_plan,
            episode_scene_id="two_block_color_sort_episode_001",
        )

    def _runtime_task_completion(self, final_state: StateSnapshotV2) -> tuple[bool, str]:
        goals = tuple((subject_ref, rule.destination.region_ref)
            for rule in self.task_plan.task.rules
            for subject_ref in rule.subjects.initial_binding.track_ids)
        return self._verifier.verify_final_region_goals(final_state, goals)

    def _hidden_truth(self, raw: RawStateSnapshot) -> MultiObjectHiddenTruthResultV1:
        return evaluate_sort(self, raw)


class MultiBlockClearRuntimeV1(BlockPickPlaceRuntimeV1):
    """Clear every initially bound block into one wide storage region."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
    ) -> None:
        self._clear_destinations = {
            "block_1": "storage_region",
            "block_2": "storage_region",
        }

        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            return build_preset_clear_region_plan_v1(
                task_id="two_block_clear_region_001",
                instruction="把工作区里的方块全部收进收纳区",
                initial_scene=scene,
                source_ref="work_region",
                destination_ref="storage_region",
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            visual_config_filename="clear_region.yaml",
            object_body_names=("block_1", "block_2"),
            task_plan_builder=build_plan,
            episode_scene_id="two_block_clear_region_episode_001",
        )

    def _runtime_task_completion(self, final_state: StateSnapshotV2) -> tuple[bool, str]:
        task = self.task_plan.task
        goals = tuple((subject_ref, task.destination.region_ref)
            for subject_ref in task.subjects.initial_binding.track_ids)
        return self._verifier.verify_final_region_goals(final_state, goals)

    def _hidden_truth(self, raw: RawStateSnapshot) -> MultiObjectHiddenTruthResultV1:
        return evaluate_clear(self, raw)


class RelativePlacementRuntimeV1(BlockPickPlaceRuntimeV1):
    """Place one block left of another via a relation-conditioned generic place."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
    ) -> None:
        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            return build_preset_relative_placement_plan_v1(
                task_id="block_left_of_block_001",
                instruction="把红色方块放到蓝色方块左边",
                initial_scene=scene,
                subject_ref="block_1",
                reference_ref="block_2",
                relation=VisualPredicate.LEFT_OF,
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            visual_config_filename="place_relative.yaml",
            object_body_names=("block_1", "block_2"),
            task_plan_builder=build_plan,
            episode_scene_id="block_left_of_block_episode_001",
        )
        initial_reference = self.backend.select_active_object("block_2")
        self._initial_reference_center = initial_reference.bowl_position_robot_base_m
        self.backend.select_active_object("block_1")

    def _hidden_truth(
        self, raw: RawStateSnapshot
    ) -> RelativePlacementHiddenTruthResultV1:
        return evaluate_relative_placement(self, raw)


class MultiBlockArrangeRuntimeV1(BlockPickPlaceRuntimeV1):
    """Arrange the initially bound blocks into a coarse, collision-free row."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
    ) -> None:
        self._arrangement_destinations = {
            "block_1": "storage_region",
            "block_2": "storage_region",
            "block_3": "storage_region",
        }

        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            return build_preset_row_arrangement_plan_v1(
                task_id="three_block_row_arrangement_001",
                instruction="把三个方块粗略排成一行并分开放置",
                initial_scene=scene,
                subject_refs=("block_1", "block_2", "block_3"),
                workspace_ref="storage_region",
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            visual_config_filename="arrange_objects.yaml",
            object_body_names=("block_1", "block_2", "block_3"),
            task_plan_builder=build_plan,
            episode_scene_id="three_block_row_arrangement_episode_001",
        )

    def _runtime_task_completion(
        self, final_state: StateSnapshotV2
    ) -> tuple[bool, str]:
        subject_refs = tuple(self._arrangement_destinations)
        scene = final_state.visual_scene
        entities = {
            entity.track_id: entity
            for entity in scene.entities
            if entity.track_id in subject_refs
        }
        if set(entities) != set(subject_refs):
            return False, "arrangement subjects are not all visible"
        if not all(
            any(
                relation.predicate is VisualPredicate.IN_REGION
                and relation.subject_ref == subject_ref
                and relation.reference_ref == "storage_region"
                and relation.value
                for relation in scene.relations
            )
            for subject_ref in subject_refs
        ):
            return False, "not all arrangement subjects are inside the workspace"
        centroids = [
            entities[subject_ref].geometry.centroid_robot_base_m.value
            for subject_ref in subject_refs
        ]
        footprints = [
            entities[subject_ref].geometry.footprint_xy_robot_base_m.value
            for subject_ref in subject_refs
        ]
        if any(value is None for value in centroids + footprints):
            return False, "arrangement geometry is unobservable"
        row_cross_span = max(value[1] for value in centroids) - min(
            value[1] for value in centroids
        )
        minimum_clearance = min(
            polygon_clearance_v1(footprints[index], footprints[other])
            for index in range(len(footprints))
            for other in range(index + 1, len(footprints))
        )
        passed = (
            row_cross_span
            <= self._verifier_config.arrangement_row_cross_axis_tolerance_m
            and minimum_clearance
            >= self._verifier_config.arrangement_minimum_pairwise_clearance_m
        )
        return (
            passed,
            "arrangement row relation passed"
            if passed
            else (
                f"arrangement row span={row_cross_span:.6f} m, "
                f"clearance={minimum_clearance:.6f} m"
            ),
        )

    def _hidden_truth(self, raw: RawStateSnapshot) -> ArrangementHiddenTruthResultV1:
        return evaluate_arrangement(self, raw)


class MultiBlockStackRuntimeV1(BlockPickPlaceRuntimeV1):
    """Stack one block on another using generic pick/transport/place capabilities."""

    _top_block_ref = "block_1"
    _base_block_ref = "block_2"

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
        object_initial_positions_world_m: Mapping[str, tuple[float, float, float]] | None = None,
    ) -> None:
        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            return build_preset_stack_plan_v1(
                task_id="two_block_stack_001",
                instruction="把红色方块堆到蓝色方块上",
                initial_scene=scene,
                subject_refs=(self._base_block_ref, self._top_block_ref),
                stacking_region_ref="table_surface",
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            visual_config_filename="stack_objects.yaml",
            object_body_names=("block_1", "block_2"),
            object_initial_positions_world_m=object_initial_positions_world_m,
            task_plan_builder=build_plan,
            episode_scene_id="two_block_stack_episode_001",
            raw_target_body_name=self._base_block_ref,
        )
        base = self.backend.select_active_object(self._base_block_ref)
        self._initial_base_center = base.bowl_position_robot_base_m
        self.backend.select_active_object(self._top_block_ref)

    def _runtime_task_completion(
        self, final_state: StateSnapshotV2
    ) -> tuple[bool, str]:
        scene = final_state.visual_scene
        on_surface = any(
            relation.predicate is VisualPredicate.ON_SURFACE
            and relation.subject_ref == self._top_block_ref
            and relation.reference_ref == self._base_block_ref
            and relation.value
            for relation in scene.relations
        )
        base_in_stacking_region = any(
            relation.predicate is VisualPredicate.IN_REGION
            and relation.subject_ref == self._base_block_ref
            and relation.reference_ref == "table_surface"
            and relation.value
            for relation in scene.relations
        )
        if not on_surface:
            return False, "top block is not visually on the base block"
        if not base_in_stacking_region:
            return False, "base block is outside the stacking region"
        return True, "stack relation passed"

    def _hidden_truth(self, raw: RawStateSnapshot) -> StackHiddenTruthResultV1:
        return evaluate_stack(self, raw)



class BlockPlaceInBowlRuntimeV1(BlockPickPlaceRuntimeV1):
    """Pick a 3 cm block and place it inside the fixed 002_bowl mesh."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
    ) -> None:
        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            return build_preset_place_inside_plan_v1(
                task_id="block_inside_bowl_001",
                instruction="把红色方块放进碗里",
                initial_scene=scene,
                subject_ref="block_1",
                container_ref="bowl_1",
                interior_region_ref="bowl_1_interior",
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            visual_config_filename="place_inside.yaml",
            object_body_names=("block_1",),
            task_plan_builder=build_plan,
            episode_scene_id="block_inside_bowl_episode_001",
            additional_support_geom_names=tuple(
                f"bowl_1_collision_{index:02d}" for index in range(24)
            ),
            raw_target_body_name="bowl_1",
            placement_obstacle_clearance_m=0.035,
        )

    def _runtime_task_completion(
        self, final_state: StateSnapshotV2
    ) -> tuple[bool, str]:
        inside = any(
            relation.predicate is VisualPredicate.INSIDE
            and relation.subject_ref == "block_1"
            and relation.reference_ref == "bowl_1_interior"
            and relation.value
            for relation in final_state.visual_scene.relations
        )
        return inside, (
            "bowl inside relation passed"
            if inside
            else "block is not fully inside the bowl interior"
        )

    def _hidden_truth(
        self, raw: RawStateSnapshot
    ) -> ContainerPlacementHiddenTruthResultV1:
        return evaluate_inside_bowl(self, raw)


class LargeButtonPressRuntimeV1(BlockPickPlaceRuntimeV1):
    """Press a large vertical button through the generic press capability."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
    ) -> None:
        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            return build_preset_press_button_plan_v1(
                task_id="large_button_press_001",
                instruction="按下黄色大按钮",
                initial_scene=scene,
                button_ref="button_1",
                press_point_ref="button_1_press_point",
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            visual_config_filename="press_button.yaml",
            object_body_names=("button_1",),
            task_plan_builder=build_plan,
            episode_scene_id="large_button_press_episode_001",
            additional_effector_geom_names=("press_tip_geom",),
        )
        initial = self.backend.snapshot()
        self._initial_button_center_xy = initial.bowl_position_robot_base_m[:2]
        self._initial_button_center_z = initial.bowl_position_robot_base_m[2]

    def _hidden_truth(self, raw: RawStateSnapshot) -> ButtonPressHiddenTruthResultV1:
        return evaluate_button_press(self, raw)


class OrderedPlaceThenPushRuntimeV1(BlockPickPlaceRuntimeV1):
    """Execute an ordered place-then-push plan using existing capabilities."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
    ) -> None:
        self._ordered_destinations = {
            "block_1": "red_region",
            "block_2": "sequence_push_region",
        }

        def build_plan(scene: VisualSceneV1) -> TaskPlanV1:
            return build_preset_place_then_push_plan_v1(
                task_id="ordered_place_then_push_001",
                instruction="先把红色方块放入红区，再把蓝色方块推入右侧宽目标区",
                initial_scene=scene,
                place_subject_ref="block_1",
                place_destination_ref="red_region",
                push_subject_ref="block_2",
                push_destination_ref="sequence_push_region",
            )

        super().__init__(
            repository_root,
            frame_sink=frame_sink,
            video_fps=video_fps,
            event_sink=event_sink,
            visual_config_filename="ordered_sequence.yaml",
            object_body_names=("block_1", "block_2"),
            task_plan_builder=build_plan,
            episode_scene_id="ordered_place_then_push_episode_001",
        )

    def _runtime_task_completion(
        self, final_state: StateSnapshotV2
    ) -> tuple[bool, str]:
        relations = final_state.visual_scene.relations
        missing = [
            subject_ref
            for subject_ref, target_ref in self._ordered_destinations.items()
            if not any(
                relation.predicate is VisualPredicate.IN_REGION
                and relation.subject_ref == subject_ref
                and relation.reference_ref == target_ref
                and relation.value
                for relation in relations
            )
        ]
        return (
            not missing,
            "ordered sequence global goal passed"
            if not missing
            else f"ordered sequence missing final goals: {', '.join(missing)}",
        )

    def _hidden_truth(
        self, raw: RawStateSnapshot
    ) -> OrderedSequenceHiddenTruthResultV1:
        return evaluate_ordered_sequence(self, raw)
