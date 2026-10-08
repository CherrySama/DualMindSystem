"""依据公开 StateSnapshot 验证能力契约，维护连续稳定观测历史。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1,
    CapabilityExecutionStateV1,
    CapabilityInstanceV1,
    CapabilityProgramV1,
    VerificationOutcome,
)
from jev4mujoco.runtime.transitions import (
    active_capability_parts_v1,
    active_skill_state_v2,
)
from jev4mujoco.contracts.state_snapshot import (
    EffectorContactMode,
    GraspState,
    StateSnapshotV2,
    SkillStateV2,
    PhaseFeedbackV1,
    PhaseMeasurementV1,
    PhaseConditionV1,
    PhaseConstraintFeedbackV1,
    PhaseEvidenceBasisV1,
    PhaseObservationBasisV1,
)
from jev4mujoco.runtime.grasp_geometry import grasp_pose_correction_v1
from jev4mujoco.runtime.progress import pick_xy_error_v1
from jev4mujoco.perception.surface_geometry import (
    footprint_clear_of_region_v1,
    polygon_contains_polygon_v1,
    signed_point_polygon_clearance_v1,
)
from jev4mujoco.contracts.visual_scene import (
    EntityV1,
    RegionV1,
    Visibility,
    VisualPredicate,
)

from jev4mujoco.runtime.verifier_config import RuntimeVerifierConfigV1
from jev4mujoco.planning.phase_definitions import (
    PHASE_DEFINITIONS_V1, GOAL_DEFINITIONS_V1, PHASE_CONDITION_SPECS_V1,
    PHASE_CONSTRAINT_SPECS_V1,
)

@dataclass(frozen=True, slots=True)
class VerificationResultV1:
    contract_id: str
    outcome: VerificationOutcome
    reason: str
    based_on_observation_id: int
    based_on_state_id: int
    based_on_action_epoch: int
    based_on_skill: SkillStateV2
    consecutive_pass_count: int = 0
    required_consecutive_passes: int = 1


class RuntimeVerifierV1:
    """Evaluates named capability contracts from public StateSnapshotV2 only."""

    def __init__(self, config: RuntimeVerifierConfigV1) -> None:
        self._config = config
        self._placement_history: dict[
            tuple[str, str], tuple[int, int, tuple[float, float, float], float]
        ] = {}
        self._surface_push_history: dict[
            tuple[str, str], tuple[int, int, tuple[float, float, float], float]
        ] = {}
        self._grasp_confirmation_started_s: dict[str, float] = {}

    def verify_final_region_goals(
        self, snapshot: StateSnapshotV2, goals: tuple[tuple[str, str], ...],
    ) -> tuple[bool, str]:
        """Recheck every initial task binding against the current public observation."""
        if not goals:
            return False, "final region verification requires bound subjects"
        if (not snapshot.freshness.visual_is_fresh
            or snapshot.freshness.visual_age_s > self._config.max_visual_age_s):
            return False, "final region goals require a fresh visual observation"
        missing = []
        for subject_ref, target_ref in goals:
            entity = self._entity(snapshot, subject_ref)
            region = self._region(snapshot, target_ref)
            if (entity is None or entity.visibility is not Visibility.CLEAR
                or entity.geometry.footprint_xy_robot_base_m.value is None):
                missing.append(f"{subject_ref} -> {target_ref}: subject geometry is unobservable")
            elif (region is None or region.geometry.boundary_xy_robot_base_m is None
                or region.geometry.boundary_xy_robot_base_m.value is None):
                missing.append(f"{subject_ref} -> {target_ref}: target boundary is unobservable")
            elif not any(
                relation.predicate is VisualPredicate.IN_REGION
                and relation.subject_ref == subject_ref
                and relation.reference_ref == target_ref
                and relation.value
                for relation in snapshot.visual_scene.relations
            ):
                missing.append(f"{subject_ref} -> {target_ref}: final in_region is not true")
        return (False, "; ".join(missing)) if missing else (True, "all bound final region goals passed")

    def verify_entry(
        self,
        instance: CapabilityInstanceV1,
        snapshot: StateSnapshotV2,
    ) -> VerificationResultV1:
        contract_id = CAPABILITY_TEMPLATES_V1[instance.kind].entry_contract
        if not snapshot.freshness.visual_is_fresh or (
            snapshot.freshness.visual_age_s > self._config.max_visual_age_s
        ):
            return self._pending(contract_id, snapshot, "fresh visual state required")
        if contract_id == "subject_visible_and_supported_v1":
            entity = self._entity(snapshot, instance.subject_ref)
            passed = (
                entity is not None
                and entity.visibility is Visibility.CLEAR
                and self._is_supported(instance, snapshot)
                and snapshot.physical.support_state.subject_ref == instance.subject_ref
            )
            return self._boolean(
                contract_id,
                snapshot,
                passed,
                "subject is visible and physically supported",
                "subject is not both visible and physically supported",
            )
        if contract_id == "subject_visible_supported_on_surface_v1":
            entity = self._entity(snapshot, instance.subject_ref)
            support = snapshot.physical.support_state
            passed = (
                entity is not None
                and entity.visibility is Visibility.CLEAR
                and support.supported
                and support.subject_ref == instance.subject_ref
                and support.support_region_ref == instance.source_ref
                and snapshot.physical.grasp_state is GraspState.NOT_HELD
                and instance.goal is not None
                and instance.goal.reference_ref
                in {
                    region.region_id for region in snapshot.visual_scene.regions
                }
            )
            return self._boolean(
                contract_id,
                snapshot,
                passed,
                "push subject is visible, supported, and goal-bound",
                "push subject is not ready on its support surface",
            )
        if contract_id == "button_visible_v1":
            entity = self._entity(snapshot, instance.subject_ref)
            point = self._region(snapshot, instance.target_ref)
            passed = (
                entity is not None
                and entity.visibility is Visibility.CLEAR
                and point is not None
                and point.owner_entity_ref == instance.subject_ref
                and point.geometry.point_robot_base_m is not None
                and point.geometry.point_robot_base_m.value is not None
            )
            return self._boolean(
                contract_id,
                snapshot,
                passed,
                "button and owned press point are visible",
                "button or owned press point is unavailable",
            )
        if contract_id == "subject_held_v1":
            return self._held_result(contract_id, instance, snapshot)
        if contract_id == "subject_held_above_region_v1":
            held = self._is_held(instance, snapshot)
            goal_satisfied = self._transport_goal_satisfied(instance, snapshot)
            passed = held and goal_satisfied
            return self._boolean(
                contract_id,
                snapshot,
                passed,
                "held subject satisfies the placement relation",
                "subject is not held at the placement relation",
                failed=not held,
            )
        return self._failed(contract_id, snapshot, "unknown entry contract")

    def verify_active_phase(
        self,
        program: CapabilityProgramV1,
        execution: CapabilityExecutionStateV1,
        snapshot: StateSnapshotV2,
    ) -> VerificationResultV1:
        _, instance, step_state, template = active_capability_parts_v1(
            program, execution
        )
        expected_skill = active_skill_state_v2(program, execution)
        if (
            snapshot.skill.skill_instance_id != expected_skill.skill_instance_id
            or snapshot.skill.skill_kind != expected_skill.skill_kind
            or snapshot.skill.phase != expected_skill.phase
            or snapshot.skill.status is not expected_skill.status
        ):
            phase = template.phases[step_state.phase_index]
            return self._failed(
                phase.completion_contract,
                snapshot,
                "snapshot skill state does not match capability execution state",
            )

        phase = template.phases[step_state.phase_index]
        contract_id = phase.completion_contract
        if phase.requires_fresh_visual and (
            not snapshot.freshness.visual_is_fresh
            or snapshot.freshness.visual_age_s > self._config.max_visual_age_s
        ):
            return self._pending(contract_id, snapshot, "fresh visual state required")

        handler = getattr(self, f"_verify_{contract_id}", None)
        if handler is None:
            return self._failed(contract_id, snapshot, "unknown phase contract")
        return handler(instance, snapshot)

    def _verify_gripper_open_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        del instance
        gripper = snapshot.robot.gripper
        passed = (
            abs(gripper.opening_m - self._config.gripper_opening_m)
            <= self._config.gripper_tolerance_m
            and abs(gripper.target_opening_m - self._config.gripper_opening_m)
            <= self._config.gripper_tolerance_m
        )
        return self._boolean(
            "gripper_open_v1", snapshot, passed, "gripper is open",
            "gripper has not reached the configured open position",
        )

    def decision_feedback(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2,
        result: VerificationResultV1, previous: PhaseFeedbackV1 | None = None,
    ) -> PhaseFeedbackV1:
        """Project the result already evaluated; never re-run stable-history checks."""
        phase = next((item for item in CAPABILITY_TEMPLATES_V1[instance.kind].phases
                      if item.phase_id == snapshot.skill.phase), None)
        skill = snapshot.skill
        goal = None if skill.goal is None else (
            skill.goal.predicate, skill.goal.reference_ref, skill.goal.desired_value)
        expected_goal = None if instance.goal is None else (
            instance.goal.predicate, instance.goal.reference_ref, instance.goal.desired_value)
        if (phase is None or result.contract_id != phase.completion_contract
            or result.based_on_observation_id != snapshot.visual_scene.observation_id
            or result.based_on_state_id != snapshot.state_id
            or result.based_on_action_epoch != snapshot.action_epoch
            or result.based_on_skill != snapshot.skill
            or (instance.instance_id, instance.kind.value, instance.subject_ref,
                instance.source_ref, instance.target_ref, expected_goal)
            != (skill.skill_instance_id, skill.skill_kind, skill.subject_ref,
                skill.source_ref, skill.target_ref, goal)):
            raise ValueError("phase feedback requires the current bound phase and observation")
        definition = PHASE_DEFINITIONS_V1[result.contract_id]
        evidence_basis = self._phase_evidence_basis(snapshot)
        parameters = []
        goal_parameters = (() if instance.goal is None else
                           GOAL_DEFINITIONS_V1[instance.goal.predicate.value].parameter_refs)
        for name in dict.fromkeys((*definition.parameter_refs, *goal_parameters)):
            value = self._config
            for part in name.split("."):
                value = getattr(value, part)
            parameters.append((name, value))
        same_phase = (previous is not None
            and previous.skill_instance_id == instance.instance_id
            and previous.phase == snapshot.skill.phase
            and previous.contract_id == result.contract_id
            and PhaseObservationBasisV1.same_scope(previous.based_on_skill, skill)
            and previous.based_on_state_id < snapshot.state_id
            and previous.based_on_action_epoch < snapshot.action_epoch
            and snapshot.recent_action is not None
            and snapshot.recent_action.start_action_epoch == previous.based_on_action_epoch
            and snapshot.recent_action.end_action_epoch == snapshot.action_epoch)
        previous_basis = None if previous is None else previous.observation_basis
        if same_phase and previous_basis is not None:
            same_phase = (
                previous_basis.scene_id == snapshot.visual_scene.scene_id
                and previous_basis.sensor_id == snapshot.visual_scene.sensor.sensor_id
                and previous_basis.calibration_id == snapshot.visual_scene.sensor.calibration_id
                and previous_basis.timestamp_s <= snapshot.recent_action.end_timestamp_s <= snapshot.timestamp_s)
        measurements = self._phase_measurements(instance, snapshot, result.contract_id)
        if result.contract_id in ("place_goal_stable_v1", "surface_push_goal_stable_v1"):
            measurements += (
                PhaseMeasurementV1("stable_observation_count", result.consecutive_pass_count, "count", "runtime_observation_history"),
                PhaseMeasurementV1("required_stable_observations", result.required_consecutive_passes, "count", "deployment_configuration"),
            )
        observation_basis = PhaseObservationBasisV1(
            snapshot.state_id, snapshot.action_epoch, snapshot.visual_scene.observation_id,
            snapshot.timestamp_s, snapshot.visual_scene.scene_id,
            snapshot.visual_scene.sensor.sensor_id, snapshot.visual_scene.sensor.calibration_id,
            snapshot.skill, evidence_basis,
            tuple((item.name, item.value is not None
                and all(value is not None for value in (item.value if isinstance(item.value, tuple) else (item.value,)))
                and self._measurement_evidence_current(item, snapshot, evidence_basis)) for item in measurements),
            None if snapshot.recent_action is None else snapshot.recent_action.end_timestamp_s,
        )
        values = {item.name: item.value for item in measurements}
        targets = {**values, **dict(parameters)}
        by_name = {item.name: item for item in measurements}
        conditions = tuple(self._condition_feedback(spec, by_name, targets, snapshot, evidence_basis)
            for spec in PHASE_CONDITION_SPECS_V1[result.contract_id])
        constraints = self._constraint_feedback(
            definition.maintain, instance, snapshot, result, conditions, by_name, targets, evidence_basis)
        return PhaseFeedbackV1(
            skill_instance_id=instance.instance_id, phase=snapshot.skill.phase,
            contract_id=result.contract_id, based_on_state_id=snapshot.state_id,
            based_on_action_epoch=snapshot.action_epoch,
            based_on_observation_id=result.based_on_observation_id,
            outcome=result.outcome.value, reason=result.reason,
            measurements=measurements,
            conditions=conditions,
            maintain_constraints=constraints,
            evidence_basis=evidence_basis,
            based_on_skill=result.based_on_skill,
            observation_basis=observation_basis,
            previous_observation_basis=previous_basis if same_phase else None,
            parameters=tuple(parameters),
            previous_measurements=previous.measurements if same_phase else (),
            previous_state_id=previous.based_on_state_id if same_phase else None,
            previous_action_epoch=previous.based_on_action_epoch if same_phase else None,
            consecutive_pass_count=result.consecutive_pass_count,
            required_consecutive_passes=result.required_consecutive_passes,
        )

    def _phase_evidence_basis(self, snapshot: StateSnapshotV2) -> PhaseEvidenceBasisV1:
        # 使用已有视觉时效配置；动作后证据还必须在该动作结束后取得。
        action_end = 0.0 if snapshot.recent_action is None else snapshot.recent_action.end_timestamp_s
        return PhaseEvidenceBasisV1(
            snapshot.sources.visual_capture_timestamp_s,
            snapshot.sources.robot_measurement_timestamp_s,
            snapshot.freshness.visual_is_fresh
            and snapshot.freshness.visual_age_s <= self._config.max_visual_age_s
            and snapshot.sources.visual_capture_timestamp_s >= action_end,
            snapshot.physical.evidence.robot_feedback_available
            # 当前同步 StateBuilder 的快照时刻就是该次机器人采样时刻。
            and snapshot.sources.robot_measurement_timestamp_s == snapshot.timestamp_s
            and snapshot.sources.robot_measurement_timestamp_s >= action_end,
        )

    @staticmethod
    def _measurement_evidence_current(measurement, snapshot, basis) -> bool:
        source = measurement.source
        if source == "unknown":
            return False
        if source in ("command", "deployment_configuration", "runtime_observation_history"):
            return True
        if source == "robot_feedback":
            return basis.robot_evidence_current
        if measurement.name in (
            "subject_supported", "subject_held", "grasp_candidate", "grasp_not_held",
            "supported_push_contact", "tool_subject_contact_clear",
        ):
            return basis.robot_evidence_current and (
                not (snapshot.physical.evidence.fresh_visual_required or source.startswith("rgbd"))
                or basis.visual_evidence_current)
        if source.startswith("rgbd"):
            return basis.visual_evidence_current and (
                "robot" not in source or basis.robot_evidence_current)
        # 现有接触/持有/支撑估计来自同一物理快照；其视觉依赖沿用证据声明。
        return basis.robot_evidence_current and (
            not snapshot.physical.evidence.fresh_visual_required or basis.visual_evidence_current)

    def _condition_feedback(self, spec, measurements, targets, snapshot, basis) -> PhaseConditionV1:
        measurement = measurements.get(spec.measurement)
        return PhaseConditionV1(
            spec.measurement, None if measurement is None else measurement.value, spec.operator,
            targets.get(spec.target) if isinstance(spec.target, str) else spec.target,
            None if spec.tolerance_parameter is None else targets.get(spec.tolerance_parameter),
            measurement is not None and self._measurement_evidence_current(measurement, snapshot, basis),
        )

    def _constraint_feedback(
        self, names, instance, snapshot, result, completion_conditions, measurements, targets, basis,
    ) -> tuple[PhaseConstraintFeedbackV1, ...]:
        feedback = []
        for name in names:
            spec = PHASE_CONSTRAINT_SPECS_V1[name]
            if spec.condition is None:
                feedback.append(PhaseConstraintFeedbackV1(name, spec.scope, "unsupported",
                    "no registered verifier covers this complete constraint; no safety certification"))
                continue
            if (spec.scope == "until_observed_release"
                and result.contract_id == "release_candidate_v1"
                and result.outcome is VerificationOutcome.PASSED
                and all(item.satisfied is True for item in completion_conditions)):
                feedback.append(PhaseConstraintFeedbackV1(name, spec.scope, "not_applicable",
                    "the current Runtime release contract passed with observed open gripper and no holding"))
                continue
            condition = self._condition_feedback(spec.condition, measurements, targets, snapshot, basis)
            evidence = measurements.get(spec.condition.measurement)
            reason = self._constraint_evidence_issue(name, instance, snapshot, evidence)
            if reason is not None:
                condition = replace(condition, evidence_current=False)
            status = {True: "satisfied", False: "violated", None: "unknown"}[condition.satisfied]
            feedback.append(PhaseConstraintFeedbackV1(name, spec.scope, status,
                reason or ("constraint evidence is missing, stale or predates the latest action"
                    if status == "unknown" else "current evidence satisfies the constraint"
                    if status == "satisfied" else "current evidence does not satisfy the constraint"),
                condition, () if evidence is None else (evidence,)))
        return tuple(feedback)

    def _constraint_evidence_issue(self, name, instance, snapshot, measurement) -> str | None:
        if measurement is None or measurement.source == "unknown":
            return "constraint measurement or its evidence source is unknown"
        physical = snapshot.physical
        if name == "subject_supported" and physical.support_state.subject_ref != instance.subject_ref:
            return "support evidence is not bound to the current subject"
        if (name == "candidate_grasp" and physical.grasp_state is GraspState.CANDIDATE_HELD
            and physical.effector_contact.subject_ref != instance.subject_ref):
            return "candidate grasp evidence is not bound to the current subject"
        if name == "press_point_alignment":
            point = self._region(snapshot, instance.target_ref)
            if point is None or point.owner_entity_ref != instance.subject_ref:
                return "press point evidence is not owned by the current subject"
        if name == "placement_relation_until_release" or measurement.source.startswith("rgbd"):
            subject = self._entity(snapshot, instance.subject_ref)
            if subject is None or subject.visibility is not Visibility.CLEAR:
                return "constraint requires a clearly observed current subject"
        return None

    def _phase_measurements(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2, contract: str,
    ) -> tuple[PhaseMeasurementV1, ...]:
        values: list[PhaseMeasurementV1] = []

        def add(name, value, unit="boolean", source="state_estimate"):
            values.append(PhaseMeasurementV1(name, value, unit, source))

        physical = snapshot.physical
        add("subject_held", None if physical.grasp_state is GraspState.UNKNOWN
            else self._is_held(instance, snapshot), source=physical.evidence.grasp_source)
        add("subject_supported", None if physical.support_state.supported is None else self._is_supported(instance, snapshot),
            source=physical.evidence.support_source)
        add("visual_follow_confirmed", physical.evidence.visual_follow_confirmed, source="rgbd_robot_feedback")
        add("grasp_not_held", None if physical.grasp_state is GraspState.UNKNOWN else
            physical.grasp_state is GraspState.NOT_HELD, source=physical.evidence.grasp_source)
        add("grasp_candidate", None if physical.grasp_state is GraspState.UNKNOWN else
            self._is_grasp_candidate(instance, snapshot),
            source=physical.evidence.grasp_source)
        contact = physical.effector_contact
        add("tool_subject_contact_clear", None if contact.mode is EffectorContactMode.UNKNOWN
            else contact.mode is EffectorContactMode.CLEAR, source=physical.evidence.contact_source)
        add("supported_push_contact", None if contact.mode is EffectorContactMode.UNKNOWN
            else contact.mode is EffectorContactMode.PUSH_CONTACT and contact.subject_ref == instance.subject_ref,
            source=physical.evidence.contact_source)
        if instance.goal is not None:
            relation = next((item for item in snapshot.visual_scene.relations
                if item.subject_ref == instance.subject_ref
                and item.reference_ref == instance.goal.reference_ref
                and item.predicate is instance.goal.predicate), None)
            add("final_goal_relation", None if relation is None else relation.value, source="rgbd_relation")
            goal_satisfied = None if relation is None else relation.value is instance.goal.desired_value
            if instance.goal.predicate is VisualPredicate.CLEAR_OF_REGION:
                entity = self._entity(snapshot, instance.subject_ref)
                region = self._region(snapshot, instance.goal.reference_ref)
                observable = (entity is not None and entity.geometry.footprint_xy_robot_base_m.value is not None
                    and region is not None and region.geometry.boundary_xy_robot_base_m is not None
                    and region.geometry.boundary_xy_robot_base_m.value is not None)
                goal_satisfied = self._surface_push_goal_satisfied(instance, snapshot) if observable else None
            add("capability_goal_satisfied", goal_satisfied, source="rgbd_relation")
        if contract in ("gripper_open_v1", "push_effector_ready_v1", "release_candidate_v1", "grasp_candidate_v1"):
            gripper = snapshot.robot.gripper
            add("gripper_position_m", gripper.opening_m, "m", "robot_feedback")
            add("gripper_commanded_position_m", gripper.target_opening_m, "m", "command")
        if contract == "tcp_subject_xy_aligned_v1":
            error = pick_xy_error_v1(snapshot, instance.subject_ref)
            add("tcp_to_subject_top_xy_m", None if error is None else
                (error.delta_x_m, error.delta_y_m, 0.0), "m", "rgbd_robot_geometry")
            add("tcp_subject_xy_error_m", None if error is None else error.distance_m, "m", "rgbd_robot_geometry")
        if contract == "tcp_grasp_pose_aligned_v1":
            correction = grasp_pose_correction_v1(snapshot, instance.subject_ref)
            add("tcp_to_subject_grasp_pose_m", correction, "m", "rgbd_robot_geometry")
            add("tcp_grasp_xy_error_m", None if correction is None else hypot(correction[0], correction[1]), "m", "rgbd_robot_geometry")
            add("tcp_grasp_height_band_error_m", None if correction is None else abs(correction[2]), "m", "rgbd_robot_geometry")
        if contract in ("grasp_probe_lifted_v1", "grasp_confirmed_v1"):
            add("grasp_probe_lifted", physical.evidence.grasp_probe_lifted, source="rgbd_robot_feedback")
            add("grasp_probe_failed", physical.evidence.grasp_probe_failed, source="rgbd_robot_feedback")
        if contract == "transport_clearance_v1":
            geometry = self._transport_clearance_geometry(instance, snapshot)
            add("subject_bottom_clearance_m", None if geometry is None else geometry[0], "m", "rgbd_geometry")
            add("required_bottom_clearance_m", None if geometry is None else geometry[1], "m", "deployment_configuration")
            add("subject_bottom_clearance_deficit_m", None if geometry is None else max(0.0, geometry[1] - geometry[0]), "m", "rgbd_geometry")
        if contract in ("held_subject_above_region_v1", "transport_confirmed_v1", "release_candidate_v1"):
            subject = self._entity(snapshot, instance.subject_ref)
            observable = subject is not None and subject.geometry.footprint_xy_robot_base_m.value is not None
            if instance.goal is None:
                observable = False
            elif instance.goal.predicate is VisualPredicate.INSIDE:
                region = self._region(snapshot, instance.goal.reference_ref)
                observable = (observable and region is not None
                    and region.geometry.boundary_xy_robot_base_m is not None
                    and region.geometry.boundary_xy_robot_base_m.value is not None)
            elif instance.goal.predicate is VisualPredicate.ON_SURFACE:
                reference = self._entity(snapshot, instance.goal.reference_ref)
                observable = (observable and reference is not None
                    and reference.geometry.footprint_xy_robot_base_m.value is not None)
            else:
                observable = observable and any(
                    relation.predicate is instance.goal.predicate
                    and relation.subject_ref == instance.subject_ref
                    and relation.reference_ref == instance.goal.reference_ref
                    for relation in snapshot.visual_scene.relations)
            add("placement_horizontal_relation_satisfied", self._transport_goal_satisfied(instance, snapshot) if observable else None,
                source="rgbd_relation")
        if contract == "release_candidate_v1":
            bottom = self._bottom_z(snapshot, instance.subject_ref)
            height = self._placement_support_height(snapshot, instance)
            add("subject_bottom_to_target_support_clearance_m", None if bottom is None or height is None else bottom - height,
                "m", "rgbd_geometry")
        if contract in ("push_precontact_aligned_v1", "push_contact_established_v1", "surface_push_goal_reached_v1", "push_effector_retracted_v1", "surface_push_goal_stable_v1"):
            geometry = self._surface_push_geometry(instance, snapshot)
            add("tcp_outside_subject_footprint_clearance_m", None if geometry is None else geometry[0], "m", "rgbd_robot_geometry")
            add("tcp_to_push_height_z_m", None if geometry is None else geometry[1]
                + self._config.surface_push_tcp_height_above_surface_m - snapshot.robot.tcp_position_robot_base_m[2], "m", "rgbd_robot_geometry")
        if contract in ("press_point_aligned_v1", "button_pressed_v1"):
            region = self._region(snapshot, instance.target_ref)
            point = None if region is None or region.geometry.point_robot_base_m is None else region.geometry.point_robot_base_m.value
            tcp = snapshot.robot.tcp_position_robot_base_m
            delta = None if point is None else (point[0] - tcp[0], point[1] - tcp[1], 0.0)
            add("tcp_to_press_point_xy_m", delta, "m", "rgbd_robot_geometry")
            add("tcp_press_point_xy_error_m", self._press_point_xy_error(instance, snapshot), "m", "rgbd_robot_geometry")
        return tuple(values)

    def _press_point_xy_error(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2,
    ) -> float | None:
        region = self._region(snapshot, instance.target_ref)
        point = (
            None
            if region is None or region.geometry.point_robot_base_m is None
            else region.geometry.point_robot_base_m.value
        )
        if point is None:
            return None
        tcp = snapshot.robot.tcp_position_robot_base_m
        return hypot(point[0] - tcp[0], point[1] - tcp[1])

    def _verify_press_point_aligned_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        error = self._press_point_xy_error(instance, snapshot)
        if error is None:
            return self._pending(
                "press_point_aligned_v1", snapshot, "press point is unobservable"
            )
        return self._boolean(
            "press_point_aligned_v1",
            snapshot,
            error <= self._config.press_point_xy_tolerance_m,
            "TCP is aligned with the press point",
            f"press point XY error {error:.6f} m exceeds tolerance",
        )

    def _verify_button_pressed_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        contact = snapshot.physical.effector_contact
        xy_error = self._press_point_xy_error(instance, snapshot)
        passed = (
            self._goal_satisfied(instance, snapshot)
            and contact.mode is EffectorContactMode.PUSH_CONTACT
            and contact.subject_ref == instance.subject_ref
            and xy_error is not None
            and xy_error <= self._config.press_point_xy_tolerance_m
        )
        return self._boolean(
            "button_pressed_v1",
            snapshot,
            passed,
            "button travel, effector contact and press-point alignment are confirmed",
            "button travel, contact or press-point alignment is not confirmed",
        )
    def _verify_tcp_subject_xy_aligned_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        error = pick_xy_error_v1(snapshot, instance.subject_ref)
        if error is None:
            return self._pending(
                "tcp_subject_xy_aligned_v1", snapshot, "subject top center is unobservable"
            )
        return self._boolean(
            "tcp_subject_xy_aligned_v1",
            snapshot,
            error.distance_m <= self._config.grasp_xy_tolerance_m,
            "TCP is aligned with subject in XY",
            f"TCP XY error {error.distance_m:.6f} m exceeds tolerance",
        )

    def _verify_tcp_grasp_pose_aligned_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        correction = grasp_pose_correction_v1(snapshot, instance.subject_ref)
        if correction is None:
            return self._pending(
                "tcp_grasp_pose_aligned_v1", snapshot, "subject grasp geometry is unobservable"
            )
        xy_error = hypot(correction[0], correction[1])
        z_error = abs(correction[2])
        passed = (
            xy_error <= self._config.grasp_xy_tolerance_m
            and z_error <= 0.0
        )
        return self._boolean(
            "tcp_grasp_pose_aligned_v1",
            snapshot,
            passed,
            "TCP is within the observed subject grasp-height band",
            f"grasp pose error is xy={xy_error:.6f} m, height-band={z_error:.6f} m",
        )

    def _verify_grasp_candidate_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        return self._boolean(
            "grasp_candidate_v1",
            snapshot,
            self._is_grasp_candidate(instance, snapshot),
            "physical evidence reports a grasp candidate",
            "physical evidence has not produced a grasp candidate",
        )

    def _verify_grasp_confirmed_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        contract = "grasp_confirmed_v1"
        if self._is_held(instance, snapshot):
            return self._held_result(contract, instance, snapshot)
        if snapshot.physical.grasp_state is not GraspState.CANDIDATE_HELD:
            return self._failed(contract, snapshot, "grasp contact was lost during confirmation")
        started = self._grasp_confirmation_started_s.setdefault(
            instance.instance_id, snapshot.timestamp_s
        )
        if snapshot.timestamp_s - started >= 0.5:
            return self._failed(contract, snapshot, "grasp was not stable after the lift probe")
        return self._pending(contract, snapshot, "waiting for a stable lifted grasp")

    def _verify_grasp_probe_lifted_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        contract = "grasp_probe_lifted_v1"
        if snapshot.physical.evidence.grasp_probe_failed:
            return self._failed(contract, snapshot, "object did not follow the lift probe")
        if snapshot.physical.grasp_state is GraspState.UNKNOWN:
            return self._pending(contract, snapshot, "grasp state is unobservable during lift probe")
        if not self._is_grasp_candidate(instance, snapshot):
            return self._failed(contract, snapshot, "grasp contact was lost during lift probe")
        return self._boolean(
            contract,
            snapshot,
            snapshot.physical.evidence.grasp_probe_lifted,
            "object visually followed the lift probe with both contacts",
            "a coupled object lift has not yet been observed",
        )

    def _verify_transport_clearance_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        if not self._is_held(instance, snapshot):
            return self._failed(
                "transport_clearance_v1", snapshot, "subject was lost during transport"
            )
        geometry = self._transport_clearance_geometry(instance, snapshot)
        if geometry is None:
            return self._pending(
                "transport_clearance_v1", snapshot, "clearance geometry is unobservable"
            )
        clearance, required_clearance = geometry
        return self._boolean(
            "transport_clearance_v1",
            snapshot,
            clearance >= required_clearance,
            "subject bottom has reached transport clearance",
            f"transport clearance {clearance:.6f} m is insufficient",
        )

    def _transport_clearance_geometry(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2,
    ) -> tuple[float, float] | None:
        bottom = self._bottom_z(snapshot, instance.subject_ref)
        source_height = self._region_height(snapshot, instance.source_ref)
        if bottom is None or source_height is None:
            return None
        support_height = source_height
        required_clearance = self._config.transport_minimum_bottom_clearance_m
        if (
            instance.goal is not None
            and instance.goal.predicate is VisualPredicate.ON_SURFACE
        ):
            target_height = self._placement_support_height(snapshot, instance)
            if target_height is None:
                return None
            support_height = max(support_height, target_height)
        elif (
            instance.goal is not None
            and instance.goal.predicate is VisualPredicate.INSIDE
        ):
            region = self._region(snapshot, instance.goal.reference_ref)
            owner = (
                None
                if region is None
                else self._entity(snapshot, region.owner_entity_ref or "")
            )
            opening_top = (
                None
                if owner is None
                else owner.geometry.top_center_robot_base_m.value
            )
            if opening_top is None:
                return None
            support_height = max(support_height, float(opening_top[2]))
            required_clearance = self._config.placement_obstacle_clearance_m
        return bottom - support_height, required_clearance

    def _verify_held_subject_above_region_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        if not self._is_held(instance, snapshot):
            return self._failed(
                "held_subject_above_region_v1",
                snapshot,
                "subject was lost during transport",
            )
        return self._boolean(
            "held_subject_above_region_v1",
            snapshot,
            self._transport_goal_satisfied(instance, snapshot),
            "held subject satisfies the placement relation",
            "held subject does not satisfy the placement relation",
        )

    def _verify_transport_confirmed_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        if not self._is_held(instance, snapshot):
            return self._failed(
                "transport_confirmed_v1", snapshot, "subject was lost during transport"
            )
        passed = (
            self._transport_goal_satisfied(instance, snapshot)
            and snapshot.physical.evidence.visual_follow_confirmed
        )
        return self._boolean(
            "transport_confirmed_v1",
            snapshot,
            passed,
            "held subject transport is visually confirmed above destination",
            "transport lacks destination alignment or visual-follow confirmation",
        )

    def _verify_release_candidate_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        del instance
        gripper = snapshot.robot.gripper
        passed = (
            snapshot.physical.grasp_state is GraspState.NOT_HELD
            and abs(gripper.opening_m - self._config.gripper_opening_m)
            <= self._config.gripper_tolerance_m
        )
        return self._boolean(
            "release_candidate_v1",
            snapshot,
            passed,
            "gripper is open and subject is no longer held",
            "release has not been physically observed",
        )

    def _verify_push_effector_ready_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        del instance
        gripper = snapshot.robot.gripper
        target = self._config.surface_push_closed_gripper_position_m
        tolerance = self._config.surface_push_gripper_position_tolerance_m
        passed = (
            abs(gripper.opening_m - target) <= tolerance
            and abs(gripper.target_opening_m - target) <= tolerance
            and snapshot.physical.grasp_state is GraspState.NOT_HELD
        )
        return self._boolean(
            "push_effector_ready_v1",
            snapshot,
            passed,
            "closed effector is ready for surface pushing",
            "effector is not closed or an object is incorrectly marked held",
        )

    def _verify_push_precontact_aligned_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        geometry = self._surface_push_geometry(instance, snapshot)
        if geometry is None:
            return self._pending(
                "push_precontact_aligned_v1",
                snapshot,
                "surface push geometry is unobservable",
            )
        clearance, support_height = geometry
        error = abs(clearance - self._config.surface_push_precontact_clearance_m)
        height_error = abs(snapshot.robot.tcp_position_robot_base_m[2]
            - support_height - self._config.surface_push_tcp_height_above_surface_m)
        contact = snapshot.physical.effector_contact
        passed = (
            clearance > 0.0
            and error <= self._config.surface_push_precontact_xy_tolerance_m
            and height_error <= self._config.surface_push_tcp_height_tolerance_m
            and contact.mode is EffectorContactMode.CLEAR
            and self._is_supported(instance, snapshot)
            and snapshot.physical.grasp_state is GraspState.NOT_HELD
        )
        return self._boolean(
            "push_precontact_aligned_v1",
            snapshot,
            passed,
            "tool is outside the supported subject at precontact clearance and height",
            f"precontact clearance error={error:.6f} m, height error={height_error:.6f} m; require support and tool separation",
        )

    def _verify_push_contact_established_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        geometry = self._surface_push_geometry(instance, snapshot)
        if geometry is None:
            return self._pending(
                "push_contact_established_v1",
                snapshot,
                "surface push geometry is unobservable",
            )
        _, support_height = geometry
        tcp_height_error = abs(
            snapshot.robot.tcp_position_robot_base_m[2]
            - (support_height + self._config.surface_push_tcp_height_above_surface_m)
        )
        contact = snapshot.physical.effector_contact
        passed = (
            contact.mode is EffectorContactMode.PUSH_CONTACT
            and contact.subject_ref == instance.subject_ref
            and self._is_supported(instance, snapshot)
            and snapshot.physical.grasp_state is GraspState.NOT_HELD
            and tcp_height_error <= self._config.surface_push_tcp_height_tolerance_m
        )
        return self._boolean(
            "push_contact_established_v1",
            snapshot,
            passed,
            "supported push contact is established",
            "push contact, support, or TCP height is not ready",
        )

    def _verify_surface_push_goal_reached_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        contact = snapshot.physical.effector_contact
        passed = (
            self._surface_push_goal_satisfied(instance, snapshot)
            and self._is_supported(instance, snapshot)
            and contact.mode is EffectorContactMode.PUSH_CONTACT
            and contact.subject_ref == instance.subject_ref
            and snapshot.physical.grasp_state is GraspState.NOT_HELD
        )
        return self._boolean(
            "surface_push_goal_reached_v1",
            snapshot,
            passed,
            "surface push goal is reached while support is maintained",
            "surface push goal has not been reached with valid contact",
        )

    def _verify_push_effector_retracted_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        passed = (
            snapshot.physical.effector_contact.mode is EffectorContactMode.CLEAR
            and self._is_supported(instance, snapshot)
            and snapshot.physical.grasp_state is GraspState.NOT_HELD
        )
        return self._boolean(
            "push_effector_retracted_v1",
            snapshot,
            passed,
            "effector is clear and pushed object remains supported",
            "effector has not cleared the supported object",
        )

    def _verify_surface_push_goal_stable_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        contract_id = "surface_push_goal_stable_v1"
        centroid = self._centroid(snapshot, instance.subject_ref)
        conditions = (
            centroid is not None
            and self._surface_push_goal_satisfied(instance, snapshot)
            and self._is_supported(instance, snapshot)
            and snapshot.physical.effector_contact.mode is EffectorContactMode.CLEAR
            and snapshot.physical.grasp_state is GraspState.NOT_HELD
        )
        return self._stable_result(
            history=self._surface_push_history,
            key=(instance.instance_id, contract_id),
            contract_id=contract_id,
            snapshot=snapshot,
            centroid=centroid,
            conditions=conditions,
            required=self._config.surface_push_required_consecutive_observations,
            minimum_interval_s=(
                self._config.surface_push_minimum_observation_interval_s
            ),
            maximum_shift_m=self._config.surface_push_maximum_centroid_shift_m,
            waiting_reason="waiting for stable pushed-object observations",
            passed_reason="surface push goal is stable for the required observations",
        )

    def _verify_place_goal_stable_v1(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> VerificationResultV1:
        contract_id = "place_goal_stable_v1"
        centroid = self._centroid(snapshot, instance.subject_ref)
        conditions = (
            centroid is not None
            and self._goal_satisfied(instance, snapshot)
            and snapshot.physical.grasp_state is GraspState.NOT_HELD
            and self._is_supported(instance, snapshot)
            and snapshot.physical.support_state.subject_ref == instance.subject_ref
        )
        return self._stable_result(
            history=self._placement_history,
            key=(instance.instance_id, contract_id),
            contract_id=contract_id,
            snapshot=snapshot,
            centroid=centroid,
            conditions=conditions,
            required=self._config.placement_required_consecutive_observations,
            minimum_interval_s=self._config.placement_minimum_observation_interval_s,
            maximum_shift_m=self._config.placement_maximum_centroid_shift_m,
            waiting_reason="waiting for consecutive stable visual observations",
            passed_reason="placement is stable for the required observations",
        )

    @staticmethod
    def _stable_result(
        *,
        history: dict[
            tuple[str, str], tuple[int, int, tuple[float, float, float], float]
        ],
        key: tuple[str, str],
        contract_id: str,
        snapshot: StateSnapshotV2,
        centroid: tuple[float, float, float] | None,
        conditions: bool,
        required: int,
        minimum_interval_s: float,
        maximum_shift_m: float,
        waiting_reason: str,
        passed_reason: str,
    ) -> VerificationResultV1:
        previous = history.get(key)
        observation_id = snapshot.visual_scene.observation_id
        if not conditions or centroid is None:
            history.pop(key, None)
            return VerificationResultV1(
                contract_id=contract_id,
                outcome=VerificationOutcome.PENDING,
                reason=waiting_reason,
                based_on_observation_id=observation_id,
                based_on_state_id=snapshot.state_id,
                based_on_action_epoch=snapshot.action_epoch,
                based_on_skill=snapshot.skill,
                consecutive_pass_count=0,
                required_consecutive_passes=required,
            )
        if previous is not None and previous[0] == observation_id:
            count = previous[1]
            last_centroid = previous[2]
        elif previous is None:
            count = 1
            last_centroid = centroid
        elif (
            snapshot.timestamp_s - previous[3] + 1.0e-12
            < minimum_interval_s
        ):
            count = previous[1]
            last_centroid = previous[2]
            return VerificationResultV1(
                contract_id=contract_id,
                outcome=VerificationOutcome.PENDING,
                reason="waiting for the configured stable-observation interval",
                based_on_observation_id=observation_id,
                based_on_state_id=snapshot.state_id,
                based_on_action_epoch=snapshot.action_epoch,
                based_on_skill=snapshot.skill,
                consecutive_pass_count=count,
                required_consecutive_passes=required,
            )
        else:
            shift = sum(
                (centroid[index] - previous[2][index]) ** 2 for index in range(3)
            ) ** 0.5
            count = (
                previous[1] + 1
                if shift <= maximum_shift_m
                else 1
            )
            last_centroid = centroid
        history[key] = (
            observation_id,
            count,
            last_centroid,
            snapshot.timestamp_s,
        )
        return VerificationResultV1(
            contract_id=contract_id,
            outcome=(
                VerificationOutcome.PASSED
                if count >= required
                else VerificationOutcome.PENDING
            ),
            reason=(
                passed_reason
                if count >= required
                else waiting_reason
            ),
            based_on_observation_id=observation_id,
            based_on_state_id=snapshot.state_id,
            based_on_action_epoch=snapshot.action_epoch,
            based_on_skill=snapshot.skill,
            consecutive_pass_count=count,
            required_consecutive_passes=required,
        )

    @staticmethod
    def _entity(snapshot: StateSnapshotV2, subject_ref: str) -> EntityV1 | None:
        return next(
            (
                entity
                for entity in snapshot.visual_scene.entities
                if entity.track_id == subject_ref
            ),
            None,
        )

    @staticmethod
    def _region(snapshot: StateSnapshotV2, region_ref: str | None) -> RegionV1 | None:
        if region_ref is None:
            return None
        return next(
            (
                region
                for region in snapshot.visual_scene.regions
                if region.region_id == region_ref
            ),
            None,
        )

    def _centroid(
        self, snapshot: StateSnapshotV2, subject_ref: str
    ) -> tuple[float, float, float] | None:
        entity = self._entity(snapshot, subject_ref)
        if entity is None or entity.visibility is not Visibility.CLEAR:
            return None
        return entity.geometry.centroid_robot_base_m.value

    def _bottom_z(self, snapshot: StateSnapshotV2, subject_ref: str) -> float | None:
        entity = self._entity(snapshot, subject_ref)
        if entity is None or entity.visibility is not Visibility.CLEAR:
            return None
        centroid = entity.geometry.centroid_robot_base_m.value
        extent = entity.geometry.extent_m.value
        if centroid is None or extent is None:
            return None
        return centroid[2] - 0.5 * extent[2]

    def _region_height(
        self, snapshot: StateSnapshotV2, region_ref: str | None
    ) -> float | None:
        region = self._region(snapshot, region_ref)
        if region is None or region.geometry.surface_height_robot_base_m is None:
            return None
        return region.geometry.surface_height_robot_base_m.value

    def _placement_support_height(
        self, snapshot: StateSnapshotV2, instance: CapabilityInstanceV1
    ) -> float | None:
        target_ref = instance.target_ref
        region_height = self._region_height(snapshot, target_ref)
        if region_height is not None:
            return region_height
        if target_ref is None:
            return None
        reference = self._entity(snapshot, target_ref)
        if reference is None:
            return None
        if (
            instance.goal is not None
            and instance.goal.predicate is VisualPredicate.ON_SURFACE
        ):
            top = reference.geometry.top_center_robot_base_m.value
            return None if top is None else top[2]
        return self._region_height(snapshot, reference.support_region_ref)

    def _surface_push_geometry(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> tuple[float, float] | None:
        if instance.goal is None or instance.goal.predicate not in (
            VisualPredicate.IN_REGION,
            VisualPredicate.CLEAR_OF_REGION,
        ):
            return None
        entity = self._entity(snapshot, instance.subject_ref)
        region = self._region(snapshot, instance.goal.reference_ref)
        if entity is None or region is None:
            return None
        footprint = entity.geometry.footprint_xy_robot_base_m.value
        boundary = region.geometry.boundary_xy_robot_base_m
        source = self._region(snapshot, instance.source_ref)
        height = None if source is None else source.geometry.surface_height_robot_base_m
        if (
            entity.visibility is not Visibility.CLEAR
            or footprint is None
            or boundary is None
            or boundary.value is None
            or height is None
            or height.value is None
        ):
            return None
        tcp = snapshot.robot.tcp_position_robot_base_m
        return signed_point_polygon_clearance_v1((tcp[0], tcp[1]), footprint), float(height.value)

    @staticmethod
    def _is_supported(instance: CapabilityInstanceV1, snapshot: StateSnapshotV2) -> bool:
        support = snapshot.physical.support_state
        return support.supported is True and support.subject_ref == instance.subject_ref

    @staticmethod
    def _is_grasp_candidate(instance: CapabilityInstanceV1, snapshot: StateSnapshotV2) -> bool:
        return (snapshot.physical.grasp_state is GraspState.CANDIDATE_HELD
            or RuntimeVerifierV1._is_held(instance, snapshot))

    @staticmethod
    def _is_held(
        instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> bool:
        return (
            snapshot.physical.grasp_state is GraspState.HELD
            and snapshot.physical.held_object_ref == instance.subject_ref
        )

    @staticmethod
    def _goal_satisfied(
        instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> bool:
        if instance.goal is None:
            return False
        return any(
            relation.predicate is instance.goal.predicate
            and relation.subject_ref == instance.subject_ref
            and relation.reference_ref == instance.goal.reference_ref
            and relation.value is instance.goal.desired_value
            for relation in snapshot.visual_scene.relations
        )

    def _transport_goal_satisfied(
        self, instance: CapabilityInstanceV1, snapshot: StateSnapshotV2
    ) -> bool:
        if (
            instance.goal is not None
            and instance.goal.predicate is VisualPredicate.INSIDE
        ):
            subject = self._entity(snapshot, instance.subject_ref)
            region = self._region(snapshot, instance.goal.reference_ref)
            footprint = (
                None
                if subject is None
                else subject.geometry.footprint_xy_robot_base_m.value
            )
            boundary = (
                None
                if region is None
                or region.geometry.boundary_xy_robot_base_m is None
                else region.geometry.boundary_xy_robot_base_m.value
            )
            return (
                footprint is not None
                and boundary is not None
                and polygon_contains_polygon_v1(boundary, footprint)
            )
        if (
            instance.goal is None
            or instance.goal.predicate is not VisualPredicate.ON_SURFACE
        ):
            return self._goal_satisfied(instance, snapshot)
        subject = self._entity(snapshot, instance.subject_ref)
        reference = self._entity(snapshot, instance.goal.reference_ref)
        if subject is None or reference is None:
            return False
        subject_footprint = subject.geometry.footprint_xy_robot_base_m.value
        reference_footprint = reference.geometry.footprint_xy_robot_base_m.value
        if subject_footprint is None or reference_footprint is None:
            return False
        subject_min_x = min(point[0] for point in subject_footprint)
        subject_max_x = max(point[0] for point in subject_footprint)
        subject_min_y = min(point[1] for point in subject_footprint)
        subject_max_y = max(point[1] for point in subject_footprint)
        reference_min_x = min(point[0] for point in reference_footprint)
        reference_max_x = max(point[0] for point in reference_footprint)
        reference_min_y = min(point[1] for point in reference_footprint)
        reference_max_y = max(point[1] for point in reference_footprint)
        overlap_x = max(
            0.0,
            min(subject_max_x, reference_max_x)
            - max(subject_min_x, reference_min_x),
        )
        overlap_y = max(
            0.0,
            min(subject_max_y, reference_max_y)
            - max(subject_min_y, reference_min_y),
        )
        subject_area = (subject_max_x - subject_min_x) * (
            subject_max_y - subject_min_y
        )
        overlap_fraction = (
            0.0 if subject_area <= 0.0 else overlap_x * overlap_y / subject_area
        )
        return (
            overlap_fraction
            >= self._config.visual_relation_config.support_minimum_overlap_fraction
        )

    def _surface_push_goal_satisfied(
        self,
        instance: CapabilityInstanceV1,
        snapshot: StateSnapshotV2,
    ) -> bool:
        if instance.goal is None:
            return False
        if instance.goal.predicate is not VisualPredicate.CLEAR_OF_REGION:
            return self._goal_satisfied(instance, snapshot)
        entity = self._entity(snapshot, instance.subject_ref)
        region = self._region(snapshot, instance.goal.reference_ref)
        if entity is None or region is None:
            return False
        footprint = entity.geometry.footprint_xy_robot_base_m.value
        boundary = region.geometry.boundary_xy_robot_base_m
        if footprint is None or boundary is None or boundary.value is None:
            return False
        return footprint_clear_of_region_v1(
            footprint,
            boundary.value,
            self._config.surface_push_clear_of_region_margin_m,
        ) is instance.goal.desired_value

    def _held_result(
        self,
        contract_id: str,
        instance: CapabilityInstanceV1,
        snapshot: StateSnapshotV2,
    ) -> VerificationResultV1:
        return self._boolean(
            contract_id,
            snapshot,
            self._is_held(instance, snapshot),
            "subject is physically confirmed held",
            "subject is not physically confirmed held",
        )

    @staticmethod
    def _boolean(
        contract_id: str,
        snapshot: StateSnapshotV2,
        passed: bool,
        passed_reason: str,
        pending_reason: str,
        *,
        failed: bool = False,
    ) -> VerificationResultV1:
        return VerificationResultV1(
            contract_id=contract_id,
            outcome=(
                VerificationOutcome.PASSED
                if passed
                else VerificationOutcome.FAILED
                if failed
                else VerificationOutcome.PENDING
            ),
            reason=passed_reason if passed else pending_reason,
            based_on_observation_id=snapshot.visual_scene.observation_id,
            based_on_state_id=snapshot.state_id,
            based_on_action_epoch=snapshot.action_epoch,
            based_on_skill=snapshot.skill,
        )

    @staticmethod
    def _pending(
        contract_id: str, snapshot: StateSnapshotV2, reason: str
    ) -> VerificationResultV1:
        return VerificationResultV1(
            contract_id=contract_id,
            outcome=VerificationOutcome.PENDING,
            reason=reason,
            based_on_observation_id=snapshot.visual_scene.observation_id,
            based_on_state_id=snapshot.state_id,
            based_on_action_epoch=snapshot.action_epoch,
            based_on_skill=snapshot.skill,
        )

    @staticmethod
    def _failed(
        contract_id: str, snapshot: StateSnapshotV2, reason: str
    ) -> VerificationResultV1:
        return VerificationResultV1(
            contract_id=contract_id,
            outcome=VerificationOutcome.FAILED,
            reason=reason,
            based_on_observation_id=snapshot.visual_scene.observation_id,
            based_on_state_id=snapshot.state_id,
            based_on_action_epoch=snapshot.action_epoch,
            based_on_skill=snapshot.skill,
        )
