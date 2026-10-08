"""共享观测—决策—动作闭环，负责能力推进、动作复核、恢复和资源关闭。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import ceil, hypot
from pathlib import Path
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from jev4mujoco.simulation.backend import MujocoBackend
from jev4mujoco.runtime.capability_runtime import (
    CapabilityRuntimeEventKind,
    CapabilityRuntimeUpdateV1,
    CapabilityRuntimeV1,
    PhaseVerificationRequestV1,
)
from jev4mujoco.planning.capabilities import (
    CapabilityProgramV1,
    CapabilityProgramStatus,
)
from jev4mujoco.planning.compiler import compile_task_capabilities_v1
from jev4mujoco.simulation.config import load_stage_config
from jev4mujoco.contracts.actions import (
    ActionRequest,
    ActionResult,
    ActionStatus,
    AtomicAction,
    RawStateSnapshot,
)
from jev4mujoco.simulation.executor import AtomicActionExecutor
from jev4mujoco.runtime.grasp_geometry import (
    grasp_height_band_v1,
    grasp_translation_step_v1,
)
from jev4mujoco.runtime.state_builder import (
    LiveStateConfigV2,
    LiveStateSnapshotBuilderV2,
)
from jev4mujoco.runtime.progress import (active_xy_alignment_error_v1,
    has_xy_alignment_objective_v1, projected_pick_xy_distance_v1)
from jev4mujoco.perception.fixed_rgbd import (
    MujocoFixedRgbdConfigV1,
    load_mujoco_fixed_rgbd_config_v1,
)
from jev4mujoco.policies.decisions import (
    CapabilityDecisionKind,
    CapabilityDecisionV1,
    CapabilityPolicyV1,
    RecoveryChoiceV1,
    RecoveryDecisionV1,
)
from jev4mujoco.planning.presets import build_preset_pick_and_place_plan_v1
from jev4mujoco.runtime.verifier import RuntimeVerifierV1
from jev4mujoco.runtime.tool_geometry import CalibratedFingerToolV1
from jev4mujoco.runtime.verifier_config import load_runtime_verifier_config_v1
from jev4mujoco.runtime.local_intent import LocalIntentRuntimeV1, load_local_intent_config_v1
from jev4mujoco.runtime.transitions import active_skill_state_v2
from jev4mujoco.contracts.local_intent import LocalIntentCandidateV1
from jev4mujoco.contracts.state_snapshot import DecisionFeedbackV1, StateSnapshotV2, PhaseFeedbackV1
from jev4mujoco.contracts.task_plan import TaskPlanV1
from jev4mujoco.contracts.visual_scene import VisualSceneV1, Visibility

from jev4mujoco.experiments.results import (
    BlockRuntimeEventV1,
    HiddenTruthResultV1,
    MultiObjectHiddenTruthResultV1,
    BlockRuntimeOutcomeV1,
    RuntimeEventSink,
)
from jev4mujoco.experiments.evaluation import (
    evaluate_pick_and_place,
    evaluate_multi_object_regions,
    quaternion_rotation,
)

class BlockPickPlaceRuntimeV1:
    """Shared live RGB-D capability loop with an explicitly installed policy."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        event_sink: RuntimeEventSink | None = None,
        object_initial_position_world_m: tuple[float, float, float] | None = None,
        object_initial_positions_world_m: Mapping[str, tuple[float, float, float]] | None = None,
        visual_config_filename: str = "pick_and_place.yaml",
        object_body_names: tuple[str, ...] = ("block_1",),
        task_plan_builder: Callable[[VisualSceneV1], TaskPlanV1] | None = None,
        episode_scene_id: str = "block_pick_place_episode_001",
        additional_support_geom_names: tuple[str, ...] = (),
        additional_effector_geom_names: tuple[str, ...] = (),
        raw_target_body_name: str | None = None,
        placement_obstacle_clearance_m: float | None = None,
    ) -> None:
        self._root = Path(repository_root).resolve()
        self._event_sink = event_sink
        self._events: list[BlockRuntimeEventV1] = []
        self._visual_config: MujocoFixedRgbdConfigV1 = load_mujoco_fixed_rgbd_config_v1(
            self._root / "configs" / visual_config_filename
        )
        self._episode_scene_id = episode_scene_id
        self._verifier_config = load_runtime_verifier_config_v1(
            self._root / "configs/runtime_verifier_v1.yaml"
        )
        if placement_obstacle_clearance_m is not None:
            self._verifier_config = replace(
                self._verifier_config,
                placement_obstacle_clearance_m=placement_obstacle_clearance_m,
            )
        executor_config = load_stage_config(self._root / "configs/executor_v1.yaml")
        self.config = replace(
            executor_config,
            scene_path=self._visual_config.scene_path,
            keyframe_name=self._visual_config.keyframe_name,
            bowl_body=object_body_names[0],
            target_body=raw_target_body_name,
        )
        self.backend = MujocoBackend(
            self.config,
            bowl_initial_position_world_m=object_initial_position_world_m,
            frame_sink=frame_sink,
            video_fps=video_fps,
            object_body_names=object_body_names,
            object_initial_positions_world_m=object_initial_positions_world_m,
            additional_support_geom_names=additional_support_geom_names,
            additional_effector_geom_names=additional_effector_geom_names,
        )
        self._visual_builder = self._visual_config.make_builder(
            self.backend.model,
            clear_of_region_margin_m=(
                self._verifier_config.surface_push_clear_of_region_margin_m
            ),
            relation_config=self._verifier_config.visual_relation_config,
            button_press_minimum_travel_m=(
                self._verifier_config.press_minimum_travel_m
                + self._verifier_config.press_visual_confirmation_margin_m
            ),
        )
        self._executor = AtomicActionExecutor(self.backend, self.config)
        self._tool_calibration = CalibratedFingerToolV1.from_model(self.backend.model, self.config.tcp_site)
        self._policy: CapabilityPolicyV1 | None = None
        self._local_intent_runtime = LocalIntentRuntimeV1(load_local_intent_config_v1(
            self._root / "configs/local_intent_v1.yaml"))
        self._state_builder = LiveStateSnapshotBuilderV2(
            LiveStateConfigV2(
                gripper_open_joint_m=self.config.gripper_open_joint_m,
                gripper_closed_joint_m=self.config.gripper_closed_joint_m,
                gripper_position_tolerance_m=self._verifier_config.gripper_tolerance_m,
                grasp_xy_tolerance_m=self._verifier_config.grasp_xy_tolerance_m,
                minimum_blocked_gripper_joint_m=0.003,
            ), self._tool_calibration,
        )
        self._observation_id = 0
        self._recent_action: ActionResult | None = None
        self._decision_feedback: DecisionFeedbackV1 | None = None
        self._previous_phase_feedback: PhaseFeedbackV1 | None = None
        self._pending_xy_alignment_review: tuple[DecisionFeedbackV1, ActionResult, LocalIntentCandidateV1 | None] | None = None
        self._review_phase_key: tuple[str, str] | None = None
        self._consecutive_rejections = 0
        self._consecutive_nonprogress_actions = 0
        self._progress_block_reason: str | None = None
        self._last_visual_scene = None
        self._last_capture_state_id: int | None = None
        self._closed = False
        self._maximum_retries = 5
        self._retry_count = 0
        self._recovery_observations = 0
        self._recovery_stop_reason: str | None = None

        # Reset only forwards constraints; a short settle makes physical support observable.
        self.backend.step(100)
        first_raw = self.backend.snapshot()
        first_scene = self._capture_scene(first_raw)
        visual_refs = {entity.track_id for entity in first_scene.entities}
        missing = set(object_body_names) - visual_refs
        if missing:
            raise ValueError(
                f"runtime object bodies must also be VisualScene track IDs: {sorted(missing)}"
            )
        self.task_plan = (
            build_preset_pick_and_place_plan_v1(
                task_id="block_pick_place_001",
                instruction="把红色方块放进目标区域",
                initial_scene=first_scene,
                subject_ref="block_1",
                destination_ref="target_region",
            )
            if task_plan_builder is None
            else task_plan_builder(first_scene)
        )
        self.program = compile_task_capabilities_v1(self.task_plan, first_scene)
        self._verifier = RuntimeVerifierV1(self._verifier_config)
        self._capability_runtime = CapabilityRuntimeV1(self.program, self._verifier)
        self._last_state = self._state_builder.build(
            raw=first_raw,
            visual_scene=first_scene,
            task_plan=self.task_plan,
            program=self.program,
            execution=self._capability_runtime.execution,
        )

    def _install_task_program(
        self,
        task_plan: TaskPlanV1,
        program: CapabilityProgramV1,
        *,
        gripper_position_tolerance_m: float | None = None,
    ) -> None:
        initial_scene = self._last_visual_scene
        if initial_scene is None:
            raise RuntimeError("task program requires an initial visual scene")
        self.task_plan = task_plan
        self.program = program
        self._verifier = RuntimeVerifierV1(self._verifier_config)
        self._capability_runtime = CapabilityRuntimeV1(self.program, self._verifier)
        self._state_builder = LiveStateSnapshotBuilderV2(
            LiveStateConfigV2(
                gripper_open_joint_m=self.config.gripper_open_joint_m,
                gripper_closed_joint_m=self.config.gripper_closed_joint_m,
                gripper_position_tolerance_m=(
                    self._verifier_config.gripper_tolerance_m
                    if gripper_position_tolerance_m is None
                    else gripper_position_tolerance_m
                ),
                grasp_xy_tolerance_m=self._verifier_config.grasp_xy_tolerance_m,
                minimum_blocked_gripper_joint_m=0.003,
            ), self._tool_calibration,
        )
        self._recent_action = None
        self._decision_feedback = None
        self._previous_phase_feedback = None
        self._local_intent_runtime.reset()
        self._pending_xy_alignment_review = None
        self._review_phase_key = None
        self._consecutive_rejections = 0
        self._consecutive_nonprogress_actions = 0
        self._progress_block_reason = None
        self._events.clear()
        raw = self.backend.snapshot()
        self._last_state = self._state_builder.build(
            raw=raw,
            visual_scene=initial_scene,
            task_plan=self.task_plan,
            program=self.program,
            execution=self._capability_runtime.execution,
        )

    def install_policy(self, policy: CapabilityPolicyV1) -> None:
        """Install a policy before the capability program starts running."""
        if self._capability_runtime.execution.status is not CapabilityProgramStatus.READY:
            raise RuntimeError("policy can only be installed before runtime execution")
        if self._recent_action is not None:
            raise RuntimeError("policy cannot be replaced after an action")
        self._policy = policy

    @property
    def grasp_xy_tolerance_m(self) -> float:
        return self._verifier_config.grasp_xy_tolerance_m

    def run(self, maximum_decisions: int = 140, *, maximum_retries: int = 5) -> BlockRuntimeOutcomeV1:
        if maximum_decisions <= 0:
            raise ValueError("maximum_decisions must be positive")
        if self._policy is None:
            raise RuntimeError("install a policy before runtime execution")
        if type(maximum_retries) is not int or maximum_retries < 0:
            raise ValueError("maximum_retries must be a nonnegative integer")
        self._maximum_retries = maximum_retries
        self._retry_count = 0
        self._recovery_observations = 0
        self._recovery_stop_reason = None
        decision_count = 0
        attempt_started_at_decision = 0
        blocked_reason: str | None = None
        pending_verification_key: tuple[str, str] | None = None
        pending_verification_started_s = 0.0

        def recover(snapshot: StateSnapshotV2, reason: str) -> bool:
            nonlocal blocked_reason, attempt_started_at_decision
            before = self._retry_count
            recovered = self._attempt_recovery(snapshot, reason)
            if self._retry_count > before:
                attempt_started_at_decision = decision_count
            if recovered:
                blocked_reason = None
            elif self._recovery_stop_reason is not None:
                blocked_reason = self._recovery_stop_reason
            return recovered

        self._record("run_limits", {"maximum_retries": maximum_retries,
                                   "maximum_decisions_per_attempt": maximum_decisions,
                                   "jev_stages": getattr(self._policy, "jev_stages", 1)})
        while True:
            snapshot = self._fresh_state()
            execution = self._capability_runtime.execution
            if execution.status is CapabilityProgramStatus.COMPLETED:
                goal_passed, goal_reason = self._verify_task_completion(snapshot)
                if not goal_passed:
                    blocked_reason = goal_reason
                    decision_count += 1
                    if recover(snapshot, goal_reason):
                        continue
                break
            if decision_count - attempt_started_at_decision >= maximum_decisions:
                blocked_reason = "maximum decision budget exhausted"
                if recover(snapshot, blocked_reason):
                    pending_verification_key = None
                    continue
                break
            if snapshot.skill.allowed_actions:
                pending_verification_key = None
            elif self._capability_runtime.execution.active_step_index is not None:
                phase_key = (snapshot.skill.skill_instance_id, snapshot.skill.phase)
                if phase_key != pending_verification_key:
                    pending_verification_key = phase_key
                    pending_verification_started_s = snapshot.timestamp_s
                elif (
                    snapshot.timestamp_s - pending_verification_started_s
                    >= self._verifier_config.maximum_pending_verification_s
                ):
                    blocked_reason = f"{snapshot.skill.phase} verification timed out"
                    decision_count += 1
                    if recover(snapshot, blocked_reason):
                        pending_verification_key = None
                        blocked_reason = None
                        continue
                    break
            if self._progress_block_reason is not None:
                blocked_reason = self._progress_block_reason
                decision_count += 1
                if recover(snapshot, blocked_reason):
                    continue
                break
            execution = self._capability_runtime.execution
            if execution.status is CapabilityProgramStatus.COMPLETED:
                break
            if execution.status is CapabilityProgramStatus.BLOCKED:
                blocked_reason = execution.blocked_reason or "capability runtime blocked"
                decision_count += 1
                if recover(snapshot, blocked_reason):
                    blocked_reason = None
                    continue
                break

            if execution.active_step_index is None:
                update = self._capability_runtime.observe(snapshot)
                self._record("capability_runtime", update)
                if update.execution.status is CapabilityProgramStatus.BLOCKED:
                    blocked_reason = update.reason
                    decision_count += 1
                    if recover(snapshot, blocked_reason):
                        blocked_reason = None
                        continue
                    break
                if update.event_kind is CapabilityRuntimeEventKind.ENTRY_PENDING:
                    phase_key = (snapshot.skill.skill_instance_id, "entry")
                    if phase_key != pending_verification_key:
                        pending_verification_key = phase_key
                        pending_verification_started_s = snapshot.timestamp_s
                    elif snapshot.timestamp_s - pending_verification_started_s >= self._verifier_config.maximum_pending_verification_s:
                        blocked_reason = "capability entry verification timed out"
                        decision_count += 1
                        if recover(snapshot, blocked_reason):
                            pending_verification_key = None
                            continue
                        break
                    self.backend.step(max(1, ceil(0.1 / self.backend.physics_dt_s)))
                continue

            step_state = execution.steps[execution.active_step_index]
            if step_state.status.value == "awaiting_verification":
                update = self._capability_runtime.observe(snapshot)
                self._record("capability_runtime", update)
                snapshot = self._attach_phase_feedback(snapshot, update)
                if update.execution.status is CapabilityProgramStatus.BLOCKED:
                    blocked_reason = update.reason
                    decision_count += 1
                    if recover(snapshot, blocked_reason):
                        blocked_reason = None
                        continue
                    break
                if update.event_kind.value == "phase_verification_pending":
                    wait_s = (
                        self._verifier_config.placement_minimum_observation_interval_s
                        if snapshot.skill.phase == "verify_placement"
                        else self._verifier_config.surface_push_minimum_observation_interval_s
                        if snapshot.skill.phase == "verify_push_goal"
                        else 0.02
                    )
                    self.backend.step(max(1, ceil(wait_s / self.backend.physics_dt_s)))
                continue

            update = self._capability_runtime.probe_active_phase(snapshot)
            self._record("capability_runtime", update)
            snapshot = self._attach_phase_feedback(snapshot, update)
            if update.execution.status is CapabilityProgramStatus.BLOCKED:
                blocked_reason = update.reason
                decision_count += 1
                if recover(snapshot, blocked_reason):
                    blocked_reason = None
                    continue
                break
            if update.event_kind is not CapabilityRuntimeEventKind.PHASE_VERIFICATION_PENDING:
                continue
            if not snapshot.skill.allowed_actions:
                wait_s = (
                    self._verifier_config.placement_minimum_observation_interval_s
                    if snapshot.skill.phase == "verify_placement"
                    else self._verifier_config.surface_push_minimum_observation_interval_s
                    if snapshot.skill.phase == "verify_push_goal"
                    else 0.02
                )
                self.backend.step(max(1, ceil(wait_s / self.backend.physics_dt_s)))
                continue

            decision = self._policy.decide(snapshot)
            decision_count += 1
            self._record("policy_decision", decision)
            if decision.kind is CapabilityDecisionKind.BLOCKED:
                blocked_reason = decision.reason
                # Remote interface errors are already retried by the API client.
                exchange = getattr(self._policy, "last_exchange", {})
                if not exchange.get("error") and recover(snapshot, blocked_reason):
                    continue
                break
            if decision.kind is CapabilityDecisionKind.REQUEST_VERIFICATION:
                update = self._request_verification(decision, snapshot)
                self._record("capability_runtime", update)
                continue
            if getattr(self._policy, "jev_stages", 1) == 2:
                try:
                    if (decision.local_intent_choice is None
                        or self.backend.snapshot().state_id != snapshot.state_id
                        or decision.based_on_state_id != snapshot.state_id
                        or decision.based_on_action_epoch != snapshot.action_epoch):
                        raise ValueError("two-stage action lacks a fresh local intent binding")
                    execution = self._local_intent_runtime.accept(decision.local_intent_choice, snapshot)
                    if decision.kind is CapabilityDecisionKind.REPLAN_LOCAL_INTENT:
                        self._local_intent_runtime.request_replan(decision.reason)
                        self._record("local_intent_replan_requested", decision)
                        continue
                    if decision.action is None or decision.action.value not in execution.candidate.allowed_actions:
                        raise ValueError("action violates the selected local intent")
                    continued = (snapshot.local_intent.execution is not None
                        and snapshot.local_intent.execution.status == "active")
                    self._record("local_intent_continued" if continued else "local_intent_selected", execution)
                    snapshot = replace(snapshot, local_intent=self._local_intent_runtime.context(snapshot))
                except ValueError as error:
                    blocked_reason = str(error)
                    self._record("local_intent_rejected", blocked_reason)
                    break
            review = self._review_xy_alignment_decision(decision, snapshot)
            if review is None:
                review = self._review_grasp_alignment_decision(decision, snapshot)
            if review is None:
                review = self._review_push_geometry_decision(decision, snapshot)
            if review is not None:
                self._decision_feedback = review
                self._record("decision_review", review)
                if review.status == "rejected":
                    self._consecutive_rejections += 1
                    if self._consecutive_rejections >= self._verifier_config.maximum_consecutive_rejections:
                        blocked_reason = f"{snapshot.skill.phase} decision rejected repeatedly without execution"
                        if recover(snapshot, blocked_reason):
                            continue
                        break
                    continue
                self._consecutive_rejections = 0
            result = self._execute(decision, snapshot)
            self._recent_action = result
            self._record("action_result", result)
            if (
                review is not None
                and has_xy_alignment_objective_v1(snapshot)
                and result.status is ActionStatus.COMPLETED
            ):
                context = snapshot.local_intent
                candidate = (None if context is None or context.execution is None
                    else context.execution.candidate)
                if candidate is not None and candidate.metric_kind != "tcp_xy":
                    candidate = None
                self._pending_xy_alignment_review = (review, result, candidate)
            if result.status in (ActionStatus.REJECTED, ActionStatus.FAILED, ActionStatus.TIMED_OUT):
                blocked_reason = result.reason or f"atomic action {result.status.value}"
                if recover(self._fresh_state(), blocked_reason):
                    continue
                break

        final_state = self._fresh_state()
        capability_success = (
            self._capability_runtime.execution.status is CapabilityProgramStatus.COMPLETED
        )
        task_completion_success, task_completion_reason = (
            self._verify_task_completion(final_state)
            if capability_success
            else (False, "capability program did not complete")
        )
        runtime_success = capability_success and task_completion_success
        hidden = self._hidden_truth(self.backend.snapshot())
        if runtime_success and hidden.success:
            reason = "runtime verifier and hidden MuJoCo truth both passed"
        elif blocked_reason is not None:
            reason = blocked_reason
        elif capability_success and not task_completion_success:
            reason = task_completion_reason
        elif runtime_success:
            reason = f"runtime passed but hidden truth failed: {hidden.reason}"
        else:
            reason = "capability program did not complete"
        if getattr(self._policy, "jev_stages", 1) == 2:
            termination = self._local_intent_runtime.stop(final_state, reason)
            if termination is not None:
                self._record("local_intent_terminated", termination)
            final_state = replace(final_state, local_intent=self._local_intent_runtime.context(final_state))
            self._last_state = final_state
        return BlockRuntimeOutcomeV1(
            completed=runtime_success,
            blocked=not runtime_success,
            reason=reason,
            decision_count=decision_count,
            observation_count=self._observation_id,
            runtime_verifier_success=runtime_success,
            hidden_ground_truth=hidden,
            overall_success=runtime_success and hidden.success,
            final_state=final_state,
            final_raw_state=self.backend.snapshot(),
            events=tuple(self._events),
        )

    def _attempt_recovery(self, snapshot: StateSnapshotV2, reason: str) -> bool:
        self._recovery_stop_reason = None
        self._record("recovery_trigger", {"reason": reason, "retry_count": self._retry_count,
                                          "maximum_retries": self._maximum_retries})
        if self._retry_count >= self._maximum_retries:
            self._recovery_stop_reason = f"maximum retry budget exhausted ({self._maximum_retries}); last failure: {reason}"
            self._record("recovery_stopped", self._recovery_stop_reason)
            return False
        if self._recovery_observations >= 5:
            self._recovery_stop_reason = f"recovery made no restart after five observations; last failure: {reason}"
            self._record("recovery_stopped", self._recovery_stop_reason)
            return False
        decide = getattr(self._policy, "decide_recovery", None)
        if decide is None:
            return False
        decision: RecoveryDecisionV1 = decide(snapshot, reason)
        self._record("recovery_decision", decision)
        if (
            decision.based_on_state_id != snapshot.state_id
            or decision.based_on_action_epoch != snapshot.action_epoch
            or self.backend.snapshot().state_id != snapshot.state_id
        ):
            self._record("recovery_rejected", "stale recovery decision")
            return False
        if decision.choice is RecoveryChoiceV1.STOP:
            self._recovery_stop_reason = f"JEV recovery chose stop; last failure: {reason}"
            exchange = getattr(self._policy, "last_exchange", {})
            if exchange.get("error"):
                self._recovery_stop_reason = f"recovery interface error: {exchange['error']}; last failure: {reason}"
            self._record("recovery_stopped", self._recovery_stop_reason)
            return False
        if decision.choice is RecoveryChoiceV1.OBSERVE:
            self._recovery_observations += 1
            self.backend.step(max(1, ceil(0.1 / self.backend.physics_dt_s)))
            return True
        try:
            update = (self._capability_runtime.restart_current_placement_chain(snapshot)
                      if decision.choice is RecoveryChoiceV1.RETRY_PICK else
                      self._capability_runtime.restart_current_capability(snapshot))
        except ValueError as error:
            self._record("recovery_rejected", str(error))
            self._recovery_observations += 1
            self.backend.step(max(1, ceil(0.1 / self.backend.physics_dt_s)))
            return True
        self.program = self._capability_runtime.program
        if snapshot.physical.held_object_ref is None:
            self._state_builder.reset_for_retry()
        self._retry_count += 1
        self._recovery_observations = 0
        self._recent_action = None
        self._decision_feedback = None
        self._previous_phase_feedback = None
        self._local_intent_runtime.reset()
        self._pending_xy_alignment_review = None
        self._review_phase_key = None
        self._consecutive_rejections = 0
        self._consecutive_nonprogress_actions = 0
        self._progress_block_reason = None
        self._record("capability_runtime", update)
        return True

    def _verify_task_completion(self, snapshot: StateSnapshotV2) -> tuple[bool, str]:
        passed, reason = self._runtime_task_completion(snapshot)
        self._record("task_completion_verification", {
            "based_on_state_id": snapshot.state_id,
            "based_on_observation_id": snapshot.visual_scene.observation_id,
            "passed": passed, "reason": reason,
        })
        return passed, reason

    def _runtime_task_completion(
        self, final_state: StateSnapshotV2
    ) -> tuple[bool, str]:
        del final_state
        return True, "capability program completion is the task completion contract"

    def _fresh_state(self) -> StateSnapshotV2:
        self._sync_active_object()
        raw = self.backend.snapshot()
        scene = (
            self._last_visual_scene
            if self._last_capture_state_id == raw.state_id
            else self._capture_scene(raw)
        )
        if scene is None:
            raise RuntimeError("live visual scene cache is unexpectedly empty")
        self._last_state = self._state_builder.build(
            raw=raw,
            visual_scene=scene,
            task_plan=self.task_plan,
            program=self.program,
            execution=self._capability_runtime.execution,
            recent_action=self._recent_action,
        )
        phase_key = (
            self._last_state.skill.skill_instance_id,
            self._last_state.skill.phase,
        )
        if phase_key != self._review_phase_key:
            self._review_phase_key = phase_key
            self._decision_feedback = None
            self._consecutive_rejections = 0
            self._consecutive_nonprogress_actions = 0
        self._observe_xy_alignment_progress(self._last_state)
        self._last_state = replace(
            self._last_state, decision_feedback=self._decision_feedback
        )
        self._record("state_snapshot", self._last_state)
        return self._last_state

    def _attach_phase_feedback(
        self, snapshot: StateSnapshotV2, update: CapabilityRuntimeUpdateV1,
    ) -> StateSnapshotV2:
        if update.verification is None:
            return snapshot
        feedback = self._capability_runtime.phase_feedback(
            snapshot, update.verification, self._previous_phase_feedback)
        snapshot = replace(snapshot, phase_feedback=feedback)
        if getattr(self._policy, "jev_stages", 1) == 2:
            if update.event_kind in (CapabilityRuntimeEventKind.PHASE_ADVANCED,
                CapabilityRuntimeEventKind.CAPABILITY_COMPLETED, CapabilityRuntimeEventKind.PROGRAM_COMPLETED,
                CapabilityRuntimeEventKind.BLOCKED) and feedback.outcome in ("passed", "failed"):
                next_skill = (active_skill_state_v2(self._capability_runtime.program, update.execution)
                    if update.event_kind is CapabilityRuntimeEventKind.PHASE_ADVANCED else None)
                termination = self._local_intent_runtime.close_phase(snapshot, next_skill)
                if termination is not None:
                    self._record("local_intent_terminated", termination)
            context = self._local_intent_runtime.context(snapshot)
            snapshot = replace(snapshot, local_intent=context)
            self._record("local_intent_feedback", context)
            self._observe_xy_alignment_progress(snapshot)
            snapshot = replace(snapshot, decision_feedback=self._decision_feedback)
        self._previous_phase_feedback = feedback
        self._last_state = snapshot
        self._record("phase_feedback", feedback)
        return snapshot

    def _review_xy_alignment_decision(
        self, decision: CapabilityDecisionV1, snapshot: StateSnapshotV2
    ) -> DecisionFeedbackV1 | None:
        if not has_xy_alignment_objective_v1(snapshot):
            return None
        if decision.action is None:
            raise ValueError("action decision has no atomic action")
        error = active_xy_alignment_error_v1(snapshot)
        step = (grasp_translation_step_v1(snapshot, snapshot.skill.subject_ref, decision.action,
            self.config.cartesian_step_m) if snapshot.skill.phase == "descend_to_grasp"
            else self.config.cartesian_step_m)
        projected = (
            None
            if error is None
            else projected_pick_xy_distance_v1(
                error, decision.action, step
            )
        )
        if error is None:
            status, reason = "rejected", "alignment XY geometry is unobservable"
        elif projected is None:
            status, reason = "rejected", "action has no valid XY alignment projection"
        elif error.distance_m - projected < (self._local_intent_runtime.config.minimum_progress_m
            if snapshot.local_intent is not None else self._verifier_config.minimum_pick_xy_progress_m):
            status, reason = "rejected", "action does not reduce projected XY alignment error enough"
        else:
            status, reason = "accepted", "projected XY alignment error decreases"
        return DecisionFeedbackV1(
            decision_id=decision.decision_id,
            action=decision.action.value,
            phase=snapshot.skill.phase,
            status=status,
            reason=reason,
            error_before_m=None if error is None else error.distance_m,
            projected_error_after_m=projected,
        )

    def _review_push_geometry_decision(self, decision, snapshot):
        if (snapshot.skill.skill_kind != "surface_push" or decision.action is AtomicAction.HOLD
            or snapshot.skill.phase not in ("align_precontact", "establish_contact", "push_toward_goal")):
            return None
        subject = next((e for e in snapshot.visual_scene.entities
            if e.track_id == snapshot.skill.subject_ref), None)
        reason = None
        if (subject is None or subject.visibility is not Visibility.CLEAR
            or subject.geometry.footprint_xy_robot_base_m.value is None
            or not snapshot.freshness.visual_is_fresh):
            reason = "complete fresh subject geometry is required for pushing; visible fragments are not a full footprint"
        elif snapshot.skill.phase == "establish_contact" and snapshot.robot.tool_geometry is None:
            reason = "calibrated tool and independent finger feedback are required for contact geometry"
        if reason is None:
            return None
        return DecisionFeedbackV1(decision.decision_id, decision.action.value,
            snapshot.skill.phase, "rejected", reason, None, None)

    def _review_grasp_alignment_decision(
        self, decision: CapabilityDecisionV1, snapshot: StateSnapshotV2
    ) -> DecisionFeedbackV1 | None:
        if snapshot.skill.skill_kind != "pick" or snapshot.skill.phase != "descend_to_grasp":
            return None
        if decision.action is None:
            raise ValueError("action decision has no atomic action")
        band = grasp_height_band_v1(snapshot, snapshot.skill.subject_ref)
        subject = next(
            (entity for entity in snapshot.visual_scene.entities
             if entity.track_id == snapshot.skill.subject_ref),
            None,
        )
        centroid = None if subject is None else subject.geometry.centroid_robot_base_m.value
        tcp = snapshot.robot.tcp_position_robot_base_m
        before = projected = None
        if band is None or centroid is None:
            reason = "subject grasp geometry is unobservable"
            accepted = False
        else:
            dx, dy = centroid[0] - tcp[0], centroid[1] - tcp[1]
            xy_error = hypot(dx, dy)
            if xy_error > self._verifier_config.grasp_xy_tolerance_m:
                before = xy_error
                horizontal = {
                    AtomicAction.FORWARD: (1.0, 0.0),
                    AtomicAction.BACKWARD: (-1.0, 0.0),
                    AtomicAction.LEFT: (0.0, 1.0),
                    AtomicAction.RIGHT: (0.0, -1.0),
                }
                if decision.action in horizontal:
                    step = grasp_translation_step_v1(
                        snapshot, snapshot.skill.subject_ref, decision.action,
                        self.config.cartesian_step_m,
                    )
                    sx, sy = horizontal[decision.action]
                    projected = hypot(dx - sx * step, dy - sy * step)
                accepted = (
                    projected is not None
                    and before - projected >= self._verifier_config.minimum_pick_xy_progress_m
                )
                reason = "grasp XY error decreases" if accepted else "action does not reduce grasp XY error"
            else:
                before = max(band[0] - tcp[2], tcp[2] - band[1], 0.0)
                needed = AtomicAction.UP if tcp[2] < band[0] else AtomicAction.DOWN
                accepted = decision.action is needed and before > 0.0
                if accepted:
                    step = grasp_translation_step_v1(
                        snapshot, snapshot.skill.subject_ref, decision.action,
                        self.config.cartesian_step_m,
                    )
                    next_z = tcp[2] + (step if needed is AtomicAction.UP else -step)
                    projected = max(band[0] - next_z, next_z - band[1], 0.0)
                reason = "grasp height error decreases" if accepted else "action does not reduce grasp height error"
        return DecisionFeedbackV1(
            decision_id=decision.decision_id,
            action=decision.action.value,
            phase=snapshot.skill.phase,
            status="accepted" if accepted else "rejected",
            reason=reason,
            error_before_m=before,
            projected_error_after_m=projected,
        )

    def _observe_xy_alignment_progress(self, snapshot: StateSnapshotV2) -> None:
        pending = self._pending_xy_alignment_review
        if pending is None:
            return
        if getattr(self._policy, "jev_stages", 1) == 2 and snapshot.local_intent is None:
            return  # Defer until fresh phase feedback has bound the current local intent.
        review, result, candidate = pending
        if (
            snapshot.action_epoch < result.end_action_epoch
            or snapshot.visual_scene.capture_timestamp_s < result.end_simulation_time_s
        ):
            return
        # Evaluate the executed action against its original target, even after intent/phase changes.
        if candidate is not None:
            target, tcp = candidate.target_robot_base_m, snapshot.robot.tcp_position_robot_base_m
            distance = hypot(target[0] - tcp[0], target[1] - tcp[1])
        else:
            error = active_xy_alignment_error_v1(snapshot)
            distance = None if error is None else error.distance_m
        if distance is None or review.error_before_m is None:
            self._progress_block_reason = "XY alignment progress could not be observed after action"
            self._pending_xy_alignment_review = None
            return
        progress = review.error_before_m - distance
        observed = replace(
            review,
            status="observed",
            reason=("measured TCP progress toward the original intent point; current reference validity is checked separately"
                if candidate is not None else "measured XY alignment progress after action"),
            error_after_m=distance,
            progress_m=progress,
        )
        self._decision_feedback = observed
        self._record("decision_review", observed)
        self._pending_xy_alignment_review = None
        if candidate is not None:
            return  # The intent manager owns the measured best-progress/replanning window.
        tolerance = (self._verifier_config.grasp_xy_tolerance_m
            if snapshot.skill.skill_kind == "pick" else self._verifier_config.press_point_xy_tolerance_m)
        if distance <= tolerance:
            self._consecutive_nonprogress_actions = 0
        elif progress < self._verifier_config.minimum_pick_xy_progress_m:
            self._consecutive_nonprogress_actions += 1
            if (
                self._consecutive_nonprogress_actions
                >= self._verifier_config.maximum_consecutive_nonprogress_actions
            ):
                self._progress_block_reason = "XY alignment error did not decrease after repeated actions"
        else:
            self._consecutive_nonprogress_actions = 0

    def _sync_active_object(self) -> None:
        execution = self._capability_runtime.execution
        if execution.active_step_index is not None:
            instance = self.program.steps[execution.active_step_index]
        else:
            instance = next(
                (
                    step
                    for index, step in enumerate(self.program.steps)
                    if execution.steps[index].status.value == "pending"
                ),
                self.program.steps[-1],
            )
        if instance.subject_ref in self.backend.object_body_names:
            self.backend.select_active_object(instance.subject_ref)

    def _capture_scene(self, raw: RawStateSnapshot):
        self._observation_id += 1
        scene = self.backend.capture_fixed_rgbd(
            self._visual_builder,
            scene_id=self._episode_scene_id,
            observation_id=self._observation_id,
        ).scene
        self._last_visual_scene = scene
        self._last_capture_state_id = raw.state_id
        return scene

    def _request_verification(
        self, decision: CapabilityDecisionV1, snapshot: StateSnapshotV2
    ) -> CapabilityRuntimeUpdateV1:
        return self._capability_runtime.request_phase_verification(
            PhaseVerificationRequestV1(
                request_id=decision.decision_id,
                skill_instance_id=decision.skill_instance_id,
                based_on_state_id=decision.based_on_state_id,
                based_on_action_epoch=decision.based_on_action_epoch,
            ),
            snapshot,
        )

    def _execute(self, decision: CapabilityDecisionV1, snapshot: StateSnapshotV2) -> ActionResult:
        if decision.action is None:
            raise ValueError("action decision has no atomic action")
        translation_step_m = None
        if snapshot.skill.phase == "descend_to_grasp" and decision.action in (
            AtomicAction.FORWARD, AtomicAction.BACKWARD,
            AtomicAction.LEFT, AtomicAction.RIGHT,
            AtomicAction.DOWN, AtomicAction.UP,
        ):
            translation_step_m = grasp_translation_step_v1(
                snapshot, snapshot.skill.subject_ref, decision.action,
                self.config.cartesian_step_m,
            )
        return self._executor.execute(
            ActionRequest(
                decision_id=decision.decision_id,
                based_on_state_id=decision.based_on_state_id,
                based_on_action_epoch=decision.based_on_action_epoch,
                action=decision.action,
                translation_step_m=translation_step_m,
            )
        )

    def _hidden_truth(self, raw: RawStateSnapshot) -> HiddenTruthResultV1:
        return evaluate_pick_and_place(self, raw)

    def _multi_object_region_hidden_truth(
        self, destinations: dict[str, str]
    ) -> MultiObjectHiddenTruthResultV1:
        return evaluate_multi_object_regions(self, destinations)

    @staticmethod
    def _quaternion_rotation(quaternion_wxyz: tuple[float, float, float, float]) -> NDArray[np.float64]:
        return quaternion_rotation(quaternion_wxyz)

    def _record(self, kind: str, payload: object) -> None:
        event = BlockRuntimeEventV1(kind=kind, payload=payload)
        self._events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)

    def close(self) -> None:
        if self._closed:
            return
        self._visual_builder.close()
        self.backend.close()
        self._closed = True

    def record_idle_tail(self, duration_s: float) -> None:
        """Advance physics without issuing actions so successful videos end legibly."""
        if duration_s < 0.0:
            raise ValueError("idle video tail duration must be nonnegative")
        if duration_s == 0.0:
            return
        self.backend.step(ceil(duration_s / self.backend.physics_dt_s))

    def __enter__(self) -> BlockPickPlaceRuntimeV1:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()
