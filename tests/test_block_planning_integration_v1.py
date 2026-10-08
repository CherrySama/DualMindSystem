from __future__ import annotations

import unittest
from pathlib import Path

import mujoco

from jev4mujoco.planning.capabilities import CapabilityKind
from jev4mujoco.planning.compiler import compile_pick_and_place_capabilities_v1
from jev4mujoco.perception.fixed_rgbd import load_mujoco_fixed_rgbd_config_v1
from jev4mujoco.planning.presets import build_preset_pick_and_place_plan_v1


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BlockPlanningIntegrationV1Test(unittest.TestCase):
    def test_rendered_first_frame_compiles_to_three_capabilities(self) -> None:
        config = load_mujoco_fixed_rgbd_config_v1(
            PROJECT_ROOT / "configs/pick_and_place.yaml"
        )
        model = mujoco.MjModel.from_xml_path(str(config.scene_path))
        data = mujoco.MjData(model)
        keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, config.keyframe_name
        )
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        mujoco.mj_forward(model, data)
        with config.make_builder(model) as builder:
            visual_scene = builder.capture(
                data,
                scene_id="block_pick_place_episode_001",
                observation_id=1,
            ).scene

        plan = build_preset_pick_and_place_plan_v1(
            task_id="block_pick_place_001",
            instruction="把红色方块放进目标区域",
            initial_scene=visual_scene,
            subject_ref="block_1",
            destination_ref="target_region",
        )
        program = compile_pick_and_place_capabilities_v1(plan, visual_scene)

        self.assertEqual(plan.task.subject.initial_binding.track_ids, ("block_1",))
        self.assertEqual(plan.task.destination.region_ref, "target_region")
        self.assertEqual(
            tuple(step.kind for step in program.steps),
            (
                CapabilityKind.PICK,
                CapabilityKind.TRANSPORT,
                CapabilityKind.PLACE,
            ),
        )
        self.assertEqual(program.steps[0].source_ref, "table_surface")


if __name__ == "__main__":
    unittest.main()
