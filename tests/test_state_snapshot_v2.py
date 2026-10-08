from __future__ import annotations

import json
import unittest
from dataclasses import replace

from jev4mujoco.contracts.state_snapshot import (
    ActionFeedbackV1,
    EffectorContactMode,
    EffectorObjectContactV2,
    DerivedInteractionV2,
    FreshnessStateV2,
    GraspState,
    GripperMotionState,
    GripperStateV2,
    InteractionStateV2,
    LocalGoalV2,
    PhysicalEvidenceV2,
    PhysicalStateV2,
    RobotExecutionState,
    RobotStateV2,
    SkillRunStatus,
    SkillStateV2,
    StateSnapshotV2,
    StateSourcesV2,
    SupportStateV2,
    TaskRunStatus,
    TaskStateV2,
)
from jev4mujoco.contracts.task_plan import TaskKind
from tests.test_visual_scene_v1 import scene


def snapshot() -> StateSnapshotV2:
    visual = scene()
    return StateSnapshotV2(
        state_id=8,
        action_epoch=2,
        timestamp_s=0.5,
        sources=StateSourcesV2(
            visual_scene_id=visual.scene_id,
            visual_observation_id=visual.observation_id,
            visual_capture_timestamp_s=visual.capture_timestamp_s,
            robot_measurement_timestamp_s=0.5,
            task_plan_id="pick_place_001",
            task_registry_version="task_registry_v1",
        ),
        task=TaskStateV2(
            task_id="pick_place_001",
            task_kind=TaskKind.PICK_AND_PLACE,
            status=TaskRunStatus.RUNNING,
            completed_subject_refs=(),
            pending_subject_refs=("track_001",),
            current_subject_ref="track_001",
            current_goal_ref="left_region",
        ),
        skill=SkillStateV2(
            skill_instance_id="pick_track_001_attempt_1",
            skill_kind="pick",
            phase="align_xy",
            subject_ref="track_001",
            target_ref=None,
            allowed_intents=("align",),
            allowed_actions=(
                "move_x_positive",
                "move_x_negative",
                "move_y_positive",
                "move_y_negative",
            ),
            status=SkillRunStatus.RUNNING,
        ),
        robot=RobotStateV2(
            joint_positions_rad=(0.0, 1.9, 1.6, -1.2, 0.0, 0.0),
            joint_velocities_rad_s=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            tcp_position_robot_base_m=(0.35, 0.0, 0.07),
            tcp_orientation_robot_base_wxyz=(0.707, 0.0, 0.707, 0.0),
            gripper=GripperStateV2(
                opening_m=0.04,
                target_opening_m=0.04,
                motion_state=GripperMotionState.STOPPED,
            ),
            execution_state=RobotExecutionState.IDLE,
        ),
        physical=PhysicalStateV2(
            grasp_state=GraspState.NOT_HELD,
            held_object_ref=None,
            effector_contact=EffectorObjectContactV2(
                subject_ref=None,
                mode=EffectorContactMode.UNKNOWN,
                left_finger_contact=False,
                right_finger_contact=False,
                support_contact=True,
            ),
            support_state=SupportStateV2(
                subject_ref="track_001",
                support_region_ref="table_surface",
                supported=True,
            ),
            evidence=PhysicalEvidenceV2(
                robot_feedback_available=True,
                visual_follow_confirmed=False,
                fresh_visual_required=False,
            ),
        ),
        visual_scene=visual,
        interaction=InteractionStateV2(
            subject_ref="track_001",
            reference_ref="left_region",
            local_goal=LocalGoalV2(
                goal_type="tcp_to_object_alignment",
                target_relation="aligned_xy",
            ),
            derived=DerivedInteractionV2(
                subject_from_tcp_robot_base_m=(-0.05, -0.10, -0.04),
                distance_m=0.1187,
                dominant_axis="y",
            ),
        ),
        freshness=FreshnessStateV2(
            visual_age_s=0.5,
            visual_is_fresh=True,
            prediction_steps=0,
            pending_contact_verification=False,
            pending_completion_verification=False,
        ),
        recent_action=ActionFeedbackV1(
            decision_id=2,
            action="move_x_positive",
            start_action_epoch=1,
            end_action_epoch=2,
            status="completed",
            commanded_delta_robot_base_m=(0.01, 0.0, 0.0),
            measured_delta_robot_base_m=(0.0095, 0.0, 0.0),
            end_timestamp_s=0.5,
        ),
    )


class StateSnapshotV2Test(unittest.TestCase):
    def test_internal_snapshot_keeps_full_robot_state(self) -> None:
        payload = snapshot().to_dict()
        self.assertIn("joint_positions_rad", payload["robot"])
        self.assertEqual(len(payload["robot"]["joint_positions_rad"]), 6)
        json.dumps(payload, sort_keys=True)

    def test_held_state_requires_an_object_reference(self) -> None:
        original = snapshot()
        with self.assertRaisesRegex(ValueError, "requires held_object_ref"):
            replace(
                original.physical,
                grasp_state=GraspState.HELD,
                held_object_ref=None,
            )

    def test_source_scene_identity_must_match(self) -> None:
        original = snapshot()
        with self.assertRaisesRegex(ValueError, "visual_scene_id"):
            replace(
                original,
                sources=replace(original.sources, visual_scene_id="different_scene"),
            )

    def test_recent_action_must_end_at_snapshot_epoch(self) -> None:
        original = snapshot()
        with self.assertRaisesRegex(ValueError, "recent action end epoch"):
            replace(original, action_epoch=3)


if __name__ == "__main__":
    unittest.main()
