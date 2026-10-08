from __future__ import annotations

import json
import unittest

from jev4mujoco.contracts.task_plan import (
    CompletionContractV1,
    EntitySelectorV1,
    EntitySetV1,
    InitialBindingV1,
    PickAndPlaceTaskV1,
    PlanningBasisV1,
    Quantifier,
    RegionRefV1,
    SortObjectsTaskV1,
    SortRuleV1,
    TASK_REGISTRY_V1,
    TaskKind,
    TaskPlanV1,
    make_task_plan_v1,
    validate_task_plan_v1,
)
from jev4mujoco.contracts.visual_scene import EntityType, RegionType
from tests.test_visual_scene_v1 import scene


class TaskPlanV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = scene()
        self.basis = PlanningBasisV1(
            scene_id=self.scene.scene_id,
            observation_id=self.scene.observation_id,
            capture_timestamp_s=self.scene.capture_timestamp_s,
        )
        self.red_block = EntitySetV1(
            selector=EntitySelectorV1(
                entity_type=EntityType.MOVABLE_OBJECT,
                category="block",
                color="red",
            ),
            quantifier=Quantifier.ONE,
            initial_binding=InitialBindingV1(
                scene_id=self.scene.scene_id,
                track_ids=("track_001",),
                count=1,
            ),
        )

    def test_pick_and_place_plan_is_canonical_and_matches_scene(self) -> None:
        plan = make_task_plan_v1(
            task_id="pick_place_001",
            instruction="把红色方块放进左区",
            planning_basis=self.basis,
            task=PickAndPlaceTaskV1(
                subject=self.red_block,
                destination=RegionRefV1("left_region", RegionType.REGION_2D),
            ),
        )

        validate_task_plan_v1(plan, self.scene)
        payload = plan.to_dict()
        self.assertEqual(payload["task"]["kind"], "pick_and_place")
        self.assertEqual(payload["completion"]["contract_id"], "pick_and_place_v1")
        self.assertEqual(payload["task_registry_version"], "task_registry_v1")
        json.dumps(payload, sort_keys=True)

    def test_plan_cannot_select_an_arbitrary_completion_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires completion contract"):
            TaskPlanV1(
                task_id="bad_contract",
                instruction="把红色方块放进左区",
                planning_basis=self.basis,
                task=PickAndPlaceTaskV1(
                    subject=self.red_block,
                    destination=RegionRefV1("left_region", RegionType.REGION_2D),
                ),
                completion=CompletionContractV1("push_to_region_v1"),
            )

    def test_plan_rejects_a_fabricated_initial_track_id(self) -> None:
        fabricated = EntitySetV1(
            selector=self.red_block.selector,
            quantifier=Quantifier.ONE,
            initial_binding=InitialBindingV1(
                scene_id=self.scene.scene_id,
                track_ids=("track_missing",),
                count=1,
            ),
        )
        plan = make_task_plan_v1(
            task_id="fabricated",
            instruction="把红色方块放进左区",
            planning_basis=self.basis,
            task=PickAndPlaceTaskV1(
                subject=fabricated,
                destination=RegionRefV1("left_region", RegionType.REGION_2D),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unknown entity"):
            validate_task_plan_v1(plan, self.scene)

    def test_all_matching_must_equal_complete_initial_match_set(self) -> None:
        empty = EntitySetV1(
            selector=self.red_block.selector,
            quantifier=Quantifier.ALL_MATCHING,
            initial_binding=InitialBindingV1(
                scene_id=self.scene.scene_id,
                track_ids=(),
                count=0,
            ),
        )
        plan = make_task_plan_v1(
            task_id="incomplete_sort",
            instruction="把红色物体放到左区",
            planning_basis=self.basis,
            task=SortObjectsTaskV1(
                rules=(
                    SortRuleV1(
                        rule_id="red_to_left",
                        subjects=empty,
                        destination=RegionRefV1(
                            "left_region", RegionType.REGION_2D
                        ),
                    ),
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "complete selector match set"):
            validate_task_plan_v1(plan, self.scene)

    def test_registry_contains_only_the_agreed_closed_task_set(self) -> None:
        self.assertEqual(set(TASK_REGISTRY_V1), set(TaskKind))
        self.assertEqual(len(TASK_REGISTRY_V1), 12)


if __name__ == "__main__":
    unittest.main()
