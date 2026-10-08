from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1,
    CapabilityKind,
    CapabilityExecutionStateV1,
    CapabilityProgramStatus,
    CapabilityProgramV1,
    CapabilityRunStatus,
    CapabilityStepStateV1,
    VerificationOutcome,
)
from jev4mujoco.runtime.transitions import (
    activate_next_capability_v1,
    active_capability_parts_v1,
    apply_active_verification_v1,
    block_next_capability_v1,
    initial_capability_execution_state_v1,
    request_active_phase_verification_v1,
    start_capability_program_v1,
)
from jev4mujoco.runtime.verifier import RuntimeVerifierV1, VerificationResultV1
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2, PhaseFeedbackV1


class CapabilityRuntimeEventKind(str, Enum):
    ENTRY_PENDING = "entry_pending"
    CAPABILITY_ACTIVATED = "capability_activated"
    AWAITING_PHASE_REQUEST = "awaiting_phase_request"
    PHASE_VERIFICATION_REQUESTED = "phase_verification_requested"
    PHASE_VERIFICATION_PENDING = "phase_verification_pending"
    PHASE_ADVANCED = "phase_advanced"
    CAPABILITY_COMPLETED = "capability_completed"
    PROGRAM_COMPLETED = "program_completed"
    BLOCKED = "blocked"
    RECOVERY_RESET = "recovery_reset"


@dataclass(frozen=True, slots=True)
class PhaseVerificationRequestV1:
    request_id: int
    skill_instance_id: str
    based_on_state_id: int
    based_on_action_epoch: int

    def __post_init__(self) -> None:
        if type(self.request_id) is not int or self.request_id <= 0:
            raise ValueError("request_id must be a positive integer")
        if not self.skill_instance_id.strip():
            raise ValueError("skill_instance_id must be a non-empty string")
        if type(self.based_on_state_id) is not int or self.based_on_state_id <= 0:
            raise ValueError("based_on_state_id must be a positive integer")
        if (
            type(self.based_on_action_epoch) is not int
            or self.based_on_action_epoch < 0
        ):
            raise ValueError("based_on_action_epoch must be nonnegative")


@dataclass(frozen=True, slots=True)
class CapabilityRuntimeUpdateV1:
    event_kind: CapabilityRuntimeEventKind
    execution: CapabilityExecutionStateV1
    verification: VerificationResultV1 | None
    reason: str
    recovery_kind: str | None = None


class CapabilityRuntimeV1:
    """Owns capability progress; policies may request, but cannot perform, transitions."""

    def __init__(
        self,
        program: CapabilityProgramV1,
        verifier: RuntimeVerifierV1,
    ) -> None:
        unavailable = tuple(
            step.kind.value
            for step in program.steps
            if not CAPABILITY_TEMPLATES_V1[step.kind].runtime_implemented
        )
        if unavailable:
            raise ValueError(
                "capability runtime is not implemented for: "
                + ", ".join(dict.fromkeys(unavailable))
            )
        self._program = program
        self._verifier = verifier
        self._execution = initial_capability_execution_state_v1(program)

    @property
    def program(self) -> CapabilityProgramV1:
        return self._program

    @property
    def execution(self) -> CapabilityExecutionStateV1:
        return self._execution

    def restart_current_placement_chain(
        self, snapshot: StateSnapshotV2
    ) -> CapabilityRuntimeUpdateV1:
        """Reset only the current subject's logical pick/place chain after a drop."""
        return self.restart_current_capability(snapshot, placement_only=True)

    def restart_current_capability(
        self, snapshot: StateSnapshotV2, *, placement_only: bool = False
    ) -> CapabilityRuntimeUpdateV1:
        """Restart from public evidence, preserving the scene and other subjects."""
        current = self._execution.active_step_index
        if current is None:
            current = next((index for index, step in enumerate(self._execution.steps)
                            if step.status is CapabilityRunStatus.PENDING), None)
        if current is None and self._execution.status is CapabilityProgramStatus.COMPLETED:
            current = len(self._program.steps) - 1
        if current is None:
            raise ValueError("recovery requires an active incomplete capability")
        instance = self._program.steps[current]
        placement = instance.kind in (CapabilityKind.PICK, CapabilityKind.TRANSPORT, CapabilityKind.PLACE)
        if placement_only and not placement:
            raise ValueError("only pick/transport/place capabilities can be retried")
        support = snapshot.physical.support_state
        subject = next(
            (entity for entity in snapshot.visual_scene.entities
             if entity.track_id == instance.subject_ref), None,
        )
        if (
            not snapshot.freshness.visual_is_fresh
            or subject is None
            or subject.geometry.centroid_robot_base_m.value is None
        ):
            raise ValueError("retry requires a fresh visible subject")
        held = snapshot.physical.held_object_ref == instance.subject_ref
        if held and instance.kind is CapabilityKind.PICK:
            raise ValueError("confirmed held pick must be verified rather than opened for retry")
        if instance.kind is not CapabilityKind.PRESS and not held and (
            support.subject_ref != instance.subject_ref or not support.supported
            or support.support_region_ref is None
        ):
            raise ValueError("retry requires a held subject or an observed support")
        start = current
        if held and instance.kind is CapabilityKind.PLACE and current > 0:
            previous = self._program.steps[current - 1]
            if previous.kind is CapabilityKind.TRANSPORT and previous.subject_ref == instance.subject_ref:
                start = current - 1
        while placement and not held and start > 0 and self._program.steps[start].kind is not CapabilityKind.PICK:
            previous = self._program.steps[start - 1]
            if previous.subject_ref != instance.subject_ref:
                break
            start -= 1
        if placement and not held and self._program.steps[start].kind is not CapabilityKind.PICK:
            raise ValueError("current capability has no reusable pick step")
        end = start + 1
        while placement and end < len(self._program.steps):
            candidate = self._program.steps[end]
            if candidate.subject_ref != instance.subject_ref or candidate.kind not in (
                CapabilityKind.TRANSPORT, CapabilityKind.PLACE,
            ):
                break
            end += 1
        attempt = self._execution.transition_index + 1
        program_steps = list(self._program.steps)
        state_steps = list(self._execution.steps)
        for index in range(start, end):
            previous = program_steps[index]
            new_id = f"{previous.instance_id}:retry{attempt}"
            program_steps[index] = replace(
                previous, instance_id=new_id,
                source_ref=(support.support_region_ref if not held and instance.kind is not CapabilityKind.PRESS
                            else previous.source_ref),
            )
            state_steps[index] = CapabilityStepStateV1(
                instance_id=new_id,
                status=CapabilityRunStatus.PENDING,
                phase_index=0,
            )
        self._program = replace(self._program, steps=tuple(program_steps))
        self._execution = CapabilityExecutionStateV1(
            program_id=self._program.program_id,
            status=CapabilityProgramStatus.RUNNING,
            active_step_index=None,
            steps=tuple(state_steps),
            transition_index=attempt,
        )
        return replace(self._update(
            CapabilityRuntimeEventKind.RECOVERY_RESET, None,
            f"retrying {instance.subject_ref} {instance.kind.value} from public observed state",
        ), recovery_kind=program_steps[start].kind.value)

    def observe(self, snapshot: StateSnapshotV2) -> CapabilityRuntimeUpdateV1:
        if self._execution.status is CapabilityProgramStatus.COMPLETED:
            return self._update(
                CapabilityRuntimeEventKind.PROGRAM_COMPLETED,
                None,
                "capability program is already complete",
            )
        if self._execution.status is CapabilityProgramStatus.BLOCKED:
            return self._update(
                CapabilityRuntimeEventKind.BLOCKED,
                None,
                self._execution.blocked_reason or "capability program is blocked",
            )

        if self._execution.active_step_index is None:
            pending_index = next(
                index
                for index, step in enumerate(self._execution.steps)
                if step.status is CapabilityRunStatus.PENDING
            )
            instance = self._program.steps[pending_index]
            verification = self._verifier.verify_entry(instance, snapshot)
            if verification.outcome is VerificationOutcome.PENDING:
                return self._update(
                    CapabilityRuntimeEventKind.ENTRY_PENDING,
                    verification,
                    verification.reason,
                )
            if verification.outcome is VerificationOutcome.FAILED:
                self._execution = block_next_capability_v1(
                    self._program,
                    self._execution,
                    reason=verification.reason,
                )
                return self._update(
                    CapabilityRuntimeEventKind.BLOCKED,
                    verification,
                    verification.reason,
                )
            if self._execution.status is CapabilityProgramStatus.READY:
                self._execution = start_capability_program_v1(
                    self._program, self._execution
                )
            else:
                self._execution = activate_next_capability_v1(
                    self._program, self._execution
                )
            return self._update(
                CapabilityRuntimeEventKind.CAPABILITY_ACTIVATED,
                verification,
                "entry contract passed; capability activated",
            )

        _, _, step_state, _ = active_capability_parts_v1(
            self._program, self._execution
        )
        if step_state.status is CapabilityRunStatus.RUNNING:
            return self._update(
                CapabilityRuntimeEventKind.AWAITING_PHASE_REQUEST,
                None,
                "runtime is waiting for a phase verification request",
            )
        if step_state.status is not CapabilityRunStatus.AWAITING_VERIFICATION:
            raise RuntimeError("active capability has an invalid runtime status")

        return self._verify_active_phase(snapshot)

    def probe_active_phase(
        self, snapshot: StateSnapshotV2
    ) -> CapabilityRuntimeUpdateV1:
        """Evaluate the current phase without waiting for a policy request.

        A pending result leaves the phase running so a policy may propose the
        next action. Passed and failed results remain Runtime-owned transitions.
        """
        _, _, step_state, _ = active_capability_parts_v1(
            self._program, self._execution
        )
        if step_state.status is not CapabilityRunStatus.RUNNING:
            raise ValueError("active phase is not accepting an automatic probe")
        return self._verify_active_phase(snapshot)

    def phase_feedback(
        self, snapshot: StateSnapshotV2, result: VerificationResultV1,
        previous: PhaseFeedbackV1 | None = None,
    ) -> PhaseFeedbackV1:
        instance = next(step for step in self._program.steps
            if step.instance_id == snapshot.skill.skill_instance_id)
        return self._verifier.decision_feedback(instance, snapshot, result, previous)

    def _verify_active_phase(
        self, snapshot: StateSnapshotV2
    ) -> CapabilityRuntimeUpdateV1:
        _, _, step_state, _ = active_capability_parts_v1(
            self._program, self._execution
        )

        before_step_index = self._execution.active_step_index
        before_phase_index = step_state.phase_index
        verification = self._verifier.verify_active_phase(
            self._program, self._execution, snapshot
        )
        if verification.outcome is VerificationOutcome.PENDING:
            return self._update(
                CapabilityRuntimeEventKind.PHASE_VERIFICATION_PENDING,
                verification,
                verification.reason,
            )
        if step_state.status is CapabilityRunStatus.RUNNING:
            self._execution = request_active_phase_verification_v1(
                self._program, self._execution
            )
        self._execution = apply_active_verification_v1(
            self._program,
            self._execution,
            verification.outcome,
            failure_reason=(
                verification.reason
                if verification.outcome is VerificationOutcome.FAILED
                else None
            ),
        )
        if self._execution.status is CapabilityProgramStatus.BLOCKED:
            return self._update(
                CapabilityRuntimeEventKind.BLOCKED,
                verification,
                verification.reason,
            )
        if self._execution.status is CapabilityProgramStatus.COMPLETED:
            return self._update(
                CapabilityRuntimeEventKind.PROGRAM_COMPLETED,
                verification,
                "final capability completion contract passed",
            )
        if self._execution.active_step_index is None:
            return self._update(
                CapabilityRuntimeEventKind.CAPABILITY_COMPLETED,
                verification,
                "capability complete; next entry contract must pass",
            )
        _, _, after_step_state, _ = active_capability_parts_v1(
            self._program, self._execution
        )
        if (
            self._execution.active_step_index != before_step_index
            or after_step_state.phase_index <= before_phase_index
        ):
            raise RuntimeError("phase verification produced an invalid forward transition")
        return self._update(
            CapabilityRuntimeEventKind.PHASE_ADVANCED,
            verification,
            "phase completion contract passed",
        )

    def request_phase_verification(
        self,
        request: PhaseVerificationRequestV1,
        snapshot: StateSnapshotV2,
    ) -> CapabilityRuntimeUpdateV1:
        _, instance, step_state, _ = active_capability_parts_v1(
            self._program, self._execution
        )
        if step_state.status is not CapabilityRunStatus.RUNNING:
            raise ValueError("active phase is not accepting a verification request")
        if (
            request.based_on_state_id != snapshot.state_id
            or request.based_on_action_epoch != snapshot.action_epoch
        ):
            raise ValueError("stale phase verification request")
        if request.skill_instance_id != instance.instance_id:
            raise ValueError("phase verification request targets a different skill")
        if snapshot.skill.skill_instance_id != instance.instance_id:
            raise ValueError("snapshot skill does not match active capability")
        self._execution = request_active_phase_verification_v1(
            self._program, self._execution
        )
        return self._update(
            CapabilityRuntimeEventKind.PHASE_VERIFICATION_REQUESTED,
            None,
            "runtime accepted the candidate phase-complete request",
        )

    def _update(
        self,
        event_kind: CapabilityRuntimeEventKind,
        verification: VerificationResultV1 | None,
        reason: str,
    ) -> CapabilityRuntimeUpdateV1:
        return CapabilityRuntimeUpdateV1(
            event_kind=event_kind,
            execution=self._execution,
            verification=verification,
            reason=reason,
        )
