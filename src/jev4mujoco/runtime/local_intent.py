"""从公开几何提供候选，保存模型选择，按新观测检查意图有效性。"""

from dataclasses import dataclass, replace
from math import hypot, isfinite
from pathlib import Path
import yaml

from jev4mujoco.contracts.local_intent import (
    LocalIntentCandidateV1, LocalIntentChoiceV1, LocalIntentContextV1, LocalIntentExecutionV1,
    LocalIntentHandoffV1, LocalIntentTerminationV1,
)
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2
from jev4mujoco.contracts.visual_scene import Visibility
from jev4mujoco.perception.surface_geometry import signed_point_polygon_clearance_v1
from jev4mujoco.planning.phase_definitions import bound_subtask_definition_v1
from jev4mujoco.planning.capabilities import CAPABILITY_TEMPLATES_V1, CapabilityKind
from jev4mujoco.runtime.grasp_geometry import grasp_height_band_v1
from jev4mujoco.runtime.tool_geometry import observed_contact_sides_v1, contact_geometry_v1


@dataclass(frozen=True, slots=True)
class LocalIntentConfigV1:
    minimum_progress_m: float
    maximum_actions_without_new_best: int
    maximum_local_replans: int
    maximum_reference_shift_m: float

    def __post_init__(self):
        for value in (self.minimum_progress_m, self.maximum_reference_shift_m):
            if not isfinite(value) or value <= 0:
                raise ValueError("local intent distances must be finite and positive")
        for value in (self.maximum_actions_without_new_best, self.maximum_local_replans):
            if type(value) is not int or value <= 0:
                raise ValueError("local intent limits must be positive integers")


def load_local_intent_config_v1(path: str | Path) -> LocalIntentConfigV1:
    data = yaml.safe_load(Path(path).read_text())
    if data.pop("schema", None) != "LocalIntentConfigV1":
        raise ValueError("unsupported local intent configuration")
    return LocalIntentConfigV1(**data)


def local_intent_metric_v1(candidate, snapshot):
    target = candidate.target_robot_base_m
    tcp = snapshot.robot.tcp_position_robot_base_m
    if candidate.metric_kind == "tcp_xy":
        return hypot(target[0] - tcp[0], target[1] - tcp[1])
    if candidate.metric_kind == "tcp_z":
        return abs(target[2] - tcp[2])
    if candidate.metric_kind == "contact_geometry":
        return contact_geometry_v1(snapshot, candidate.contact_side)["local_error_m"]
    return None  # A Boolean phase result is not a continuous progress estimate.


def local_intent_correction_v1(candidate, snapshot):
    target = candidate.target_robot_base_m
    tcp = snapshot.robot.tcp_position_robot_base_m
    if candidate.metric_kind == "tcp_xy":
        return (target[0] - tcp[0], target[1] - tcp[1], 0.0)
    if candidate.metric_kind == "tcp_z":
        return (0.0, 0.0, target[2] - tcp[2])
    if candidate.metric_kind == "contact_geometry":
        return contact_geometry_v1(snapshot, candidate.contact_side).get("correction_robot_base_m")
    return None


def local_intent_candidates_v1(snapshot: StateSnapshotV2, *, for_validation=False, retained_approach=None) -> tuple[LocalIntentCandidateV1, ...]:
    """Candidates specify local conditions; their order is not a route ranking."""
    skill, feedback = snapshot.skill, snapshot.phase_feedback
    if feedback is None or (not skill.allowed_actions and not for_validation):
        return ()
    definition = bound_subtask_definition_v1(skill.skill_instance_id, skill.skill_kind,
        skill.subject_ref, skill.source_ref, skill.target_ref, None if skill.goal is None else
        (skill.goal.predicate.value, skill.goal.reference_ref, skill.goal.desired_value))
    phase = next(item[2] for item in definition.phases if item[0] == skill.phase)
    parameters = dict(feedback.parameters)
    measurements = {item.name: item.value for item in feedback.measurements}
    entity = next((item for item in snapshot.visual_scene.entities if item.track_id == skill.subject_ref), None)
    target = next((item for item in snapshot.visual_scene.regions if item.region_id == skill.target_ref), None)
    tcp = snapshot.robot.tcp_position_robot_base_m
    candidates = []

    def geometric(name, objective, axes, point, tolerance, reference, side=None):
        # Geometry remains observable when Runtime changes the permitted action axes.
        available = ("forward", "backward", "left", "right", "up", "down", "hold") if for_validation else skill.allowed_actions
        actions = tuple(action for action in available if action == "hold" or
            action in ({"forward", "backward", "left", "right"} if axes == "xy" else {"up", "down"}))
        if actions:
            candidate = LocalIntentCandidateV1(name, objective, actions, "tcp_" + axes,
                point, tolerance, reference, phase.maintain, "public_rgbd_and_robot_feedback", contact_side=side)
            candidates.append(candidate)

    if feedback.contract_id == "tcp_subject_xy_aligned_v1":
        point = None if entity is None or entity.visibility is not Visibility.CLEAR else entity.geometry.top_center_robot_base_m.value
        if point is not None:
            geometric("align_subject_xy", phase.objective, "xy", point, parameters["grasp_xy_tolerance_m"], skill.subject_ref)
    elif feedback.contract_id == "tcp_grasp_pose_aligned_v1":
        point = None if entity is None or entity.visibility is not Visibility.CLEAR else entity.geometry.centroid_robot_base_m.value
        band = grasp_height_band_v1(snapshot, skill.subject_ref)
        if point is not None and band is not None:
            geometric("align_grasp_xy", "Align TCP with subject centroid in XY, preserving height.", "xy", point,
                parameters["grasp_xy_tolerance_m"], skill.subject_ref)
            if hypot(point[0] - tcp[0], point[1] - tcp[1]) <= parameters["grasp_xy_tolerance_m"]:
                geometric("adjust_grasp_height", "Reach the observed grasp-height band while preserving XY alignment.", "z",
                    (tcp[0], tcp[1], 0.5 * (band[0] + band[1])), 0.5 * (band[1] - band[0]), skill.subject_ref)
    elif feedback.contract_id == "press_point_aligned_v1":
        point = None if target is None or target.geometry.point_robot_base_m is None else target.geometry.point_robot_base_m.value
        if point is not None:
            geometric("align_press_point", phase.objective, "xy", point, parameters["press_point_xy_tolerance_m"], skill.target_ref)
    elif feedback.contract_id == "transport_clearance_v1":
        deficit = measurements.get("subject_bottom_clearance_deficit_m")
        if deficit is not None and deficit > 0:
            candidates.append(LocalIntentCandidateV1("raise_subject_clearance", phase.objective,
                tuple(action for action in skill.allowed_actions if action in ("up", "down", "hold")),
                "phase_contract", reference_ref=skill.subject_ref, maintain=phase.maintain))
    elif feedback.contract_id == "push_precontact_aligned_v1":
        footprint = None if entity is None or entity.visibility is not Visibility.CLEAR else entity.geometry.footprint_xy_robot_base_m.value
        source = next((item for item in snapshot.visual_scene.regions if item.region_id == skill.source_ref), None)
        boundary = None if source is None or source.geometry.boundary_xy_robot_base_m is None else source.geometry.boundary_xy_robot_base_m.value
        if footprint is not None and boundary is not None:
            sides = {side.edge_index: side for side in observed_contact_sides_v1(snapshot)}
            area2 = sum(footprint[i - 1][0] * point[1] - point[0] * footprint[i - 1][1] for i, point in enumerate(footprint))
            clearance = parameters["surface_push_precontact_clearance_m"]
            tolerance = parameters["surface_push_precontact_xy_tolerance_m"]
            for index, end in enumerate(footprint):
                start = footprint[index - 1]
                dx, dy = end[0] - start[0], end[1] - start[1]
                length = hypot(dx, dy)
                if length <= 1e-12 or index not in sides:
                    continue
                orientation = 1.0 if area2 > 0 else -1.0
                point = ((start[0] + end[0]) * 0.5 + orientation * dy / length * clearance,
                    (start[1] + end[1]) * 0.5 - orientation * dx / length * clearance, tcp[2])
                # Public point geometry only; no arm/tool path safety is inferred.
                if (signed_point_polygon_clearance_v1(point[:2], boundary) <= 0
                    and abs(signed_point_polygon_clearance_v1(point[:2], footprint) - clearance) <= tolerance):
                    geometric(f"approach_edge_{index}", "Approach this observed outer-side point in XY. Cross-phase side inheritance requires fresh verified reach. Preserve tool height while aligning.",
                        "xy", point, tolerance, skill.subject_ref,
                        sides[index])
        height_delta = measurements.get("tcp_to_push_height_z_m")
        if height_delta is not None:
            geometric("adjust_push_height", "Reach the configured tool height relative to source support, preserving TCP XY.", "z",
                (tcp[0], tcp[1], tcp[2] + height_delta), parameters["surface_push_tcp_height_tolerance_m"], skill.source_ref)
    elif feedback.contract_id == "push_contact_established_v1":
        sides = observed_contact_sides_v1(snapshot)
        if retained_approach is not None and retained_approach.contact_side is not None:
            sides = tuple(s for s in sides if s.edge_index == retained_approach.contact_side.edge_index)
        if snapshot.robot.tool_geometry is not None:
            for side in sides:
                candidates.append(LocalIntentCandidateV1(f"contact_edge_{side.edge_index}",
                    "Establish actual supported contact on this selected finite side; use tool gap, tangent and height overlap, and preserve support. Geometric overlap alone does not confirm contact.",
                    (skill.allowed_actions or (("forward", "backward", "left", "right", "up", "down", "hold")
                        if for_validation else ())), "contact_geometry", reference_ref=skill.subject_ref,
                    maintain=phase.maintain, geometry_source="calibrated_tool_and_public_rgbd",
                    contact_side=side))
    elif feedback.contract_id == "release_candidate_v1":
        for name, objective, allowed in (
            ("lower_to_release", "Adjust subject bottom clearance to support while preserving placement and holding the subject; reconsider this intent when release is appropriate.", ("up", "down", "hold")),
            ("release_subject", "Release the subject by opening the gripper when observed placement geometry supports release; Runtime verifies the outcome.", ("gripper_open", "hold")),
        ):
            actions = tuple(action for action in skill.allowed_actions if action in allowed)
            if actions:
                candidates.append(LocalIntentCandidateV1(name, objective, actions, "phase_contract",
                    reference_ref=skill.target_ref, maintain=phase.maintain))
    else:
        for name in skill.allowed_intents:
            candidates.append(LocalIntentCandidateV1(name, phase.objective, skill.allowed_actions,
                "phase_contract", reference_ref=skill.target_ref or skill.subject_ref, maintain=phase.maintain))
    return tuple(candidates)


class LocalIntentRuntimeV1:
    """No action/side selection occurs here; only the model's valid choice is retained."""

    def __init__(self, config: LocalIntentConfigV1):
        self.config = config
        self._next_id = 1
        self.reset()

    def reset(self):
        self._key = None
        self._execution = None
        self._retained_approach = None
        self._handoff = None
        self._handoff_reference = None
        self._last_termination = None
        self._phase_closed = False
        self._replans = 0
        self._blocked_reason = None

    def context(self, snapshot: StateSnapshotV2) -> LocalIntentContextV1:
        key = self._scope(snapshot)
        if key != self._key:
            if not self._phase_closed and self._execution is not None:
                self._terminate(snapshot, "cancelled", "scope changed without explicit phase completion", "scope_changed")
            if (self._key is None or key[0] != self._key[0] or key[2:] != self._key[2:]
                or self._handoff is None or key[1] != self._handoff.destination_phase):
                self._handoff = None
                self._handoff_reference = None
            self._retained_approach = None  # 本阶段选过的计划不直接跨阶段继承。
            self._key = key
            self._execution = None
            self._replans = 0
            self._blocked_reason = None
            self._phase_closed = False
        if self._phase_closed:
            return LocalIntentContextV1(snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id,
                snapshot.skill.skill_instance_id, snapshot.skill.phase, (), self._execution,
                self._replans, self._blocked_reason, last_termination=self._last_termination)
        if self._handoff is not None:
            reached = self._reached_approach(self._handoff_reference, snapshot)
            self._handoff = None if reached is None else replace(self._handoff, candidate=reached[0],
                metric_m=reached[1], based_on_state_id=snapshot.state_id,
                based_on_action_epoch=snapshot.action_epoch,
                based_on_observation_id=snapshot.visual_scene.observation_id)
            if self._handoff is None:
                self._handoff_reference = None
        if snapshot.skill.skill_kind == "surface_push" and not observed_contact_sides_v1(snapshot):
            self._retained_approach = None
        inherited = None if self._handoff is None else self._handoff.candidate
        candidates = local_intent_candidates_v1(snapshot, retained_approach=inherited)
        execution = self._execution
        if execution is not None and execution.status == "active":
            geometry = local_intent_candidates_v1(snapshot, for_validation=True, retained_approach=inherited)
            replacement = next((item for item in geometry if item.choice_id == execution.candidate.choice_id), None)
            metric = local_intent_metric_v1(execution.candidate, snapshot)
            reason, status = "intent remains valid", "active"
            if not snapshot.freshness.visual_is_fresh:
                status, reason = "invalidated", "fresh observation required to continue intent"
            elif (execution.candidate.metric_kind in ("phase_contract", "contact_geometry")
                and snapshot.phase_feedback is not None and snapshot.phase_feedback.outcome == "passed"):
                status, reason = "completed", "Runtime phase contract passed"
            elif replacement is None:
                status, reason = "invalidated", "intent geometry/eligibility is no longer observable"
            elif self._reference_shift(execution.candidate, replacement) > self.config.maximum_reference_shift_m:
                status, reason = "invalidated", "observed reference geometry moved beyond intent validity margin"
            elif (execution.candidate.metric_kind in ("tcp_xy", "tcp_z")
                and metric is not None and metric <= execution.candidate.tolerance_m):
                status, reason = "completed", "local geometric condition reached; phase verification remains Runtime-owned"
            elif not any(item.choice_id == execution.candidate.choice_id for item in candidates):
                status, reason = "invalidated", "current Runtime action permissions no longer permit this intent"
            count, best = execution.nonprogress_actions, execution.best_metric_m
            epoch = execution.last_observed_action_epoch
            action = snapshot.recent_action
            if (status == "active" and metric is not None and action is not None
                and snapshot.action_epoch > epoch
                and snapshot.visual_scene.capture_timestamp_s >= action.end_timestamp_s):
                epoch = snapshot.action_epoch
                if best is None or best - metric >= self.config.minimum_progress_m:
                    best, count = metric, 0
                else:
                    count += 1
                if count >= self.config.maximum_actions_without_new_best:
                    status, reason = "invalidated", "repeated actions produced no new best local geometric progress"
            self._execution = replace(execution, status=status, reason=reason, metric_m=metric,
                best_metric_m=best, nonprogress_actions=count, last_observed_action_epoch=epoch)
            if status == "invalidated":
                if execution.candidate.choice_id.startswith("approach_edge_"):
                    self._retained_approach = None
                self._replans += 1
                if self._replans >= self.config.maximum_local_replans:
                    self._blocked_reason = "local intent replanning budget exhausted; higher-level replanning required"
        if self._retained_approach is not None and snapshot.skill.phase == "align_precontact":
            geometry = local_intent_candidates_v1(snapshot, for_validation=True)
            retained = next((item for item in geometry if item.choice_id == self._retained_approach.choice_id), None)
            if (retained is None or self._reference_shift(self._retained_approach, retained)
                > self.config.maximum_reference_shift_m):
                self._retained_approach = None
            else:
                # Retain the side actually chosen by JEV; do not pick a new side on its behalf.
                candidates = tuple(item for item in candidates if not item.choice_id.startswith("approach_edge_")
                    or item.choice_id == self._retained_approach.choice_id)
        pending = tuple(item for item in candidates if item.metric_kind in ("phase_contract", "contact_geometry")
            or local_intent_metric_v1(item, snapshot) > item.tolerance_m)
        return LocalIntentContextV1(snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id,
            snapshot.skill.skill_instance_id, snapshot.skill.phase, pending, self._execution,
            self._replans, self._blocked_reason, self._retained_approach,
            self._handoff, self._last_termination)

    def close_phase(self, snapshot: StateSnapshotV2, next_skill):
        """原 Runtime 已结束阶段；结束意图不会重新裁决该阶段。"""
        feedback = snapshot.phase_feedback
        contract = next(item.completion_contract for item in
            CAPABILITY_TEMPLATES_V1[CapabilityKind(snapshot.skill.skill_kind)].phases
            if item.phase_id == snapshot.skill.phase)
        if (feedback is None or feedback.outcome not in ("passed", "failed")
            or feedback.contract_id != contract
            or (feedback.skill_instance_id, feedback.phase, feedback.based_on_state_id,
                feedback.based_on_action_epoch, feedback.based_on_observation_id)
            != (snapshot.skill.skill_instance_id, snapshot.skill.phase, snapshot.state_id,
                snapshot.action_epoch, snapshot.visual_scene.observation_id)):
            raise ValueError("phase termination requires current terminal Runtime feedback")
        key = self._scope(snapshot)
        if self._key != key:
            self.context(snapshot)
        if self._phase_closed:
            return None  # 重复附加同一反馈不能再生成结束事件。
        self._handoff = None
        self._handoff_reference = None
        if (feedback.outcome == "passed" and snapshot.skill.phase == "align_precontact"
            and next_skill is not None and next_skill.phase == "establish_contact"
            and next_skill.skill_instance_id == snapshot.skill.skill_instance_id
            and next_skill.goal == snapshot.skill.goal and next_skill.subject_ref == snapshot.skill.subject_ref
            and next_skill.source_ref == snapshot.skill.source_ref and next_skill.target_ref == snapshot.skill.target_ref
            and next_skill.skill_kind == snapshot.skill.skill_kind
            and self._retained_approach is not None):
            reached = self._reached_approach(self._retained_approach, snapshot)
            if reached is not None:
                self._handoff = LocalIntentHandoffV1(reached[0], snapshot.skill.skill_instance_id,
                    snapshot.skill.phase, next_skill.phase, snapshot.state_id, snapshot.action_epoch,
                    snapshot.visual_scene.observation_id, reached[1])
                self._handoff_reference = self._retained_approach
        termination = None
        if self._execution is not None:
            candidate = self._execution.candidate
            best = self._execution.best_metric_m
            status, reason = "cancelled", "Runtime phase failed; local intent ended"
            metric = local_intent_metric_v1(candidate, snapshot)
            if feedback.outcome == "passed":
                status, reason = "superseded", "Runtime phase passed without proving the selected local point reached"
                if candidate.metric_kind in ("phase_contract", "contact_geometry"):
                    status, reason = "completed", "corresponding actual Runtime phase contract passed"
                elif self._fresh_after_action(snapshot):
                    replacement = next((item for item in local_intent_candidates_v1(snapshot, for_validation=True)
                        if item.choice_id == candidate.choice_id), None)
                    if replacement is not None and self._reference_shift(candidate, replacement) <= self.config.maximum_reference_shift_m:
                        metric = local_intent_metric_v1(replacement, snapshot)
                        if metric is not None:
                            best = metric if best is None else min(best, metric)
                        if metric is not None and metric <= candidate.tolerance_m:
                            status, reason = "completed", "local geometric condition reached in phase-end observation"
            self._execution = replace(self._execution, metric_m=metric, best_metric_m=best)
            termination = self._terminate(snapshot, status, reason, "phase_" + feedback.outcome)
        self._retained_approach = None
        self._phase_closed = True
        return termination

    def _terminate(self, snapshot, status, reason, trigger):
        self._execution = replace(self._execution, status=status, reason=reason,
            last_observed_action_epoch=snapshot.action_epoch)
        self._last_termination = LocalIntentTerminationV1(self._execution, self._key[0], self._key[1],
            snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id, trigger)
        return self._last_termination

    def stop(self, snapshot: StateSnapshotV2, reason: str):
        """回合最终停止时结束命令，不伪造阶段失败或改变恢复路径。"""
        if self._execution is None or self._phase_closed:
            return None
        status = "cancelled" if self._execution.status == "active" else self._execution.status
        self._execution = replace(self._execution, metric_m=(
            local_intent_metric_v1(self._execution.candidate, snapshot)
            if self._fresh_after_action(snapshot) else None))
        termination = self._terminate(snapshot, status, reason, "run_stopped")
        self._retained_approach = None
        self._handoff = None
        self._handoff_reference = None
        self._phase_closed = True
        return termination

    @staticmethod
    def _scope(snapshot):
        skill = snapshot.skill
        return (skill.skill_instance_id, skill.phase, skill.goal, skill.subject_ref,
            skill.source_ref, skill.target_ref, skill.skill_kind)

    @staticmethod
    def _fresh_after_action(snapshot):
        return (snapshot.freshness.visual_is_fresh and (snapshot.recent_action is None
            or snapshot.visual_scene.capture_timestamp_s >= snapshot.recent_action.end_timestamp_s))

    def _reached_approach(self, candidate, snapshot):
        if (not self._fresh_after_action(snapshot) or candidate.metric_kind != "tcp_xy"
            or candidate.contact_side is None or candidate.reference_ref != snapshot.skill.subject_ref):
            return None
        side = next((item for item in observed_contact_sides_v1(snapshot)
            if item.edge_index == candidate.contact_side.edge_index), None)
        if side is None:
            return None
        original = candidate.contact_side
        # 原选择的外侧距离不变；用本次观测的侧面重建 XY，不能拿旧点证明已达。
        clearance = sum((candidate.target_robot_base_m[i] -
            (original.start_xy_m[i] + original.end_xy_m[i]) * 0.5) * original.outward_normal_xy[i]
            for i in (0, 1))
        point = tuple((side.start_xy_m[i] + side.end_xy_m[i]) * 0.5
            + side.outward_normal_xy[i] * clearance for i in (0, 1)) + (snapshot.robot.tcp_position_robot_base_m[2],)
        current = replace(candidate, contact_side=side, target_robot_base_m=point)
        if self._reference_shift(candidate, current) > self.config.maximum_reference_shift_m:
            return None
        metric = local_intent_metric_v1(current, snapshot)
        return (current, metric) if metric <= candidate.tolerance_m else None

    @staticmethod
    def _reference_shift(first, second):
        if first.reference_ref != second.reference_ref:
            return float("inf")
        if first.contact_side is not None and second.contact_side is not None:
            if sum(a * b for a, b in zip(first.contact_side.outward_normal_xy,
                second.contact_side.outward_normal_xy)) < 1 - 1e-6:
                return float("inf")
            return max(hypot(*(a - b for a, b in zip(old, new)))
                for old, new in ((first.contact_side.start_xy_m, second.contact_side.start_xy_m),
                    (first.contact_side.end_xy_m, second.contact_side.end_xy_m)))
        if first.target_robot_base_m is None or second.target_robot_base_m is None:
            return 0.0
        axes = (0, 1) if first.metric_kind == "tcp_xy" else (2,)
        return sum((first.target_robot_base_m[i] - second.target_robot_base_m[i]) ** 2 for i in axes) ** 0.5

    def accept(self, choice: LocalIntentChoiceV1, snapshot: StateSnapshotV2):
        context = snapshot.local_intent
        if (context is None or context.blocked_reason is not None
            or choice.based_on_state_id != snapshot.state_id
            or choice.based_on_action_epoch != snapshot.action_epoch
            or choice.skill_instance_id != snapshot.skill.skill_instance_id
            or choice.phase != snapshot.skill.phase):
            raise ValueError("stale or invalid local intent choice")
        candidate = next((item for item in context.candidates if item.choice_id == choice.choice_id), None)
        if candidate is None:
            raise ValueError("model selected an unknown local intent candidate")
        if context.execution is not None and context.execution.status == "active":
            if context.execution.candidate.choice_id != choice.choice_id:
                raise ValueError("cannot silently replace an active local intent")
            return context.execution
        metric = local_intent_metric_v1(candidate, snapshot)
        self._execution = LocalIntentExecutionV1(self._next_id, candidate, snapshot.state_id,
            snapshot.action_epoch, "active", "validated JEV-selected local intent", metric, metric, 0, snapshot.action_epoch)
        self._next_id += 1
        if candidate.choice_id.startswith("approach_edge_"):
            self._retained_approach = candidate
        return self._execution

    def request_replan(self, reason: str):
        if self._execution is None or self._execution.status != "active":
            raise ValueError("local replanning requires an active intent")
        self._execution = replace(self._execution, status="invalidated", reason=reason)
        self._replans += 1
        if self._execution.candidate.choice_id.startswith("approach_edge_"):
            self._retained_approach = None
        if self._replans >= self.config.maximum_local_replans:
            self._blocked_reason = "local intent replanning budget exhausted; higher-level replanning required"
