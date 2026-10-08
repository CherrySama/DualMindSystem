"""能力执行状态的转换与当前阶段投影；转换由 Runtime 调用。"""

from __future__ import annotations

from jev4mujoco.contracts.state_snapshot import (
    SkillGoalV1,
    SkillRunStatus,
    SkillStateV2,
)

from jev4mujoco.planning.capabilities import (
    CAPABILITY_TEMPLATES_V1,
    CapabilityRunStatus,
    CapabilityProgramStatus,
    VerificationOutcome,
    _identifier,
    CapabilityTemplateV1,
    CapabilityInstanceV1,
    CapabilityProgramV1,
    CapabilityStepStateV1,
    CapabilityExecutionStateV1,
)

def initial_capability_execution_state_v1(
    program: CapabilityProgramV1,
) -> CapabilityExecutionStateV1:
    return CapabilityExecutionStateV1(
        program_id=program.program_id,
        status=CapabilityProgramStatus.READY,
        active_step_index=None,
        steps=tuple(
            CapabilityStepStateV1(
                instance_id=step.instance_id,
                status=CapabilityRunStatus.PENDING,
                phase_index=0,
            )
            for step in program.steps
        ),
        transition_index=0,
    )


def start_capability_program_v1(
    program: CapabilityProgramV1,
    state: CapabilityExecutionStateV1,
) -> CapabilityExecutionStateV1:
    _validate_execution_matches_program(program, state)
    if state.status is not CapabilityProgramStatus.READY:
        raise ValueError("only a ready capability program can start")
    return activate_next_capability_v1(program, state)


def activate_next_capability_v1(
    program: CapabilityProgramV1,
    state: CapabilityExecutionStateV1,
) -> CapabilityExecutionStateV1:
    _validate_execution_matches_program(program, state)
    if state.status not in (
        CapabilityProgramStatus.READY,
        CapabilityProgramStatus.RUNNING,
    ):
        raise ValueError("only a ready or running capability program can activate")
    if state.active_step_index is not None:
        raise ValueError("a capability is already active")
    pending_indices = [
        index
        for index, step in enumerate(state.steps)
        if step.status is CapabilityRunStatus.PENDING
    ]
    if not pending_indices:
        raise ValueError("capability program has no pending step to activate")
    next_index = pending_indices[0]
    if any(
        step.status is not CapabilityRunStatus.COMPLETED
        for step in state.steps[:next_index]
    ):
        raise ValueError("cannot activate a capability before prior steps complete")
    steps = list(state.steps)
    steps[next_index] = CapabilityStepStateV1(
        instance_id=steps[next_index].instance_id,
        status=CapabilityRunStatus.RUNNING,
        phase_index=0,
    )
    return CapabilityExecutionStateV1(
        program_id=state.program_id,
        status=CapabilityProgramStatus.RUNNING,
        active_step_index=next_index,
        steps=tuple(steps),
        transition_index=state.transition_index + 1,
    )


def block_next_capability_v1(
    program: CapabilityProgramV1,
    state: CapabilityExecutionStateV1,
    *,
    reason: str,
) -> CapabilityExecutionStateV1:
    _validate_execution_matches_program(program, state)
    _identifier(reason, "blocked reason")
    if state.active_step_index is not None:
        raise ValueError("block_next_capability_v1 expects no active capability")
    pending_indices = [
        index
        for index, step in enumerate(state.steps)
        if step.status is CapabilityRunStatus.PENDING
    ]
    if not pending_indices:
        raise ValueError("capability program has no pending step to block")
    blocked_index = pending_indices[0]
    steps = list(state.steps)
    steps[blocked_index] = CapabilityStepStateV1(
        instance_id=steps[blocked_index].instance_id,
        status=CapabilityRunStatus.BLOCKED,
        phase_index=steps[blocked_index].phase_index,
    )
    return CapabilityExecutionStateV1(
        program_id=state.program_id,
        status=CapabilityProgramStatus.BLOCKED,
        active_step_index=blocked_index,
        steps=tuple(steps),
        transition_index=state.transition_index + 1,
        blocked_reason=reason,
    )


def request_active_phase_verification_v1(
    program: CapabilityProgramV1,
    state: CapabilityExecutionStateV1,
) -> CapabilityExecutionStateV1:
    step_index, instance, step_state, _ = active_capability_parts_v1(program, state)
    if step_state.status is not CapabilityRunStatus.RUNNING:
        raise ValueError("only a running capability phase can request verification")
    steps = list(state.steps)
    steps[step_index] = CapabilityStepStateV1(
        instance_id=instance.instance_id,
        status=CapabilityRunStatus.AWAITING_VERIFICATION,
        phase_index=step_state.phase_index,
    )
    return CapabilityExecutionStateV1(
        program_id=state.program_id,
        status=state.status,
        active_step_index=step_index,
        steps=tuple(steps),
        transition_index=state.transition_index + 1,
    )


def apply_active_verification_v1(
    program: CapabilityProgramV1,
    state: CapabilityExecutionStateV1,
    outcome: VerificationOutcome,
    *,
    failure_reason: str | None = None,
) -> CapabilityExecutionStateV1:
    step_index, instance, step_state, template = active_capability_parts_v1(
        program, state
    )
    if step_state.status is not CapabilityRunStatus.AWAITING_VERIFICATION:
        raise ValueError("active capability is not awaiting verification")
    if outcome is VerificationOutcome.PENDING:
        if failure_reason is not None:
            raise ValueError("pending verification cannot carry a failure reason")
        return state

    steps = list(state.steps)
    if outcome is VerificationOutcome.FAILED:
        _identifier(failure_reason or "", "failure_reason")
        steps[step_index] = CapabilityStepStateV1(
            instance_id=instance.instance_id,
            status=CapabilityRunStatus.BLOCKED,
            phase_index=step_state.phase_index,
        )
        return CapabilityExecutionStateV1(
            program_id=state.program_id,
            status=CapabilityProgramStatus.BLOCKED,
            active_step_index=step_index,
            steps=tuple(steps),
            transition_index=state.transition_index + 1,
            blocked_reason=failure_reason,
        )

    if failure_reason is not None:
        raise ValueError("passed verification cannot carry a failure reason")
    next_phase_index = step_state.phase_index + 1
    if next_phase_index < len(template.phases):
        steps[step_index] = CapabilityStepStateV1(
            instance_id=instance.instance_id,
            status=CapabilityRunStatus.RUNNING,
            phase_index=next_phase_index,
        )
        return CapabilityExecutionStateV1(
            program_id=state.program_id,
            status=CapabilityProgramStatus.RUNNING,
            active_step_index=step_index,
            steps=tuple(steps),
            transition_index=state.transition_index + 1,
        )

    steps[step_index] = CapabilityStepStateV1(
        instance_id=instance.instance_id,
        status=CapabilityRunStatus.COMPLETED,
        phase_index=step_state.phase_index,
    )
    next_index = step_index + 1
    if next_index == len(program.steps):
        return CapabilityExecutionStateV1(
            program_id=state.program_id,
            status=CapabilityProgramStatus.COMPLETED,
            active_step_index=None,
            steps=tuple(steps),
            transition_index=state.transition_index + 1,
        )
    return CapabilityExecutionStateV1(
        program_id=state.program_id,
        status=CapabilityProgramStatus.RUNNING,
        active_step_index=None,
        steps=tuple(steps),
        transition_index=state.transition_index + 1,
    )


def active_skill_state_v2(
    program: CapabilityProgramV1,
    state: CapabilityExecutionStateV1,
) -> SkillStateV2:
    _, instance, step_state, template = active_capability_parts_v1(program, state)
    phase = template.phases[step_state.phase_index]
    status_map = {
        CapabilityRunStatus.RUNNING: SkillRunStatus.RUNNING,
        CapabilityRunStatus.AWAITING_VERIFICATION: SkillRunStatus.AWAITING_VERIFICATION,
        CapabilityRunStatus.COMPLETED: SkillRunStatus.COMPLETED,
        CapabilityRunStatus.BLOCKED: SkillRunStatus.BLOCKED,
    }
    if step_state.status is CapabilityRunStatus.PENDING:
        raise ValueError("pending capability cannot be projected as the active skill")
    return SkillStateV2(
        skill_instance_id=instance.instance_id,
        skill_kind=instance.kind.value,
        phase=phase.phase_id,
        subject_ref=instance.subject_ref,
        target_ref=instance.target_ref,
        source_ref=instance.source_ref,
        allowed_intents=(
            phase.allowed_intents
            if step_state.status is CapabilityRunStatus.RUNNING
            else ()
        ),
        allowed_actions=(
            tuple(action.value for action in phase.allowed_actions)
            if step_state.status is CapabilityRunStatus.RUNNING
            else ()
        ),
        status=status_map[step_state.status],
        goal=(
            None
            if instance.goal is None
            else SkillGoalV1(
                predicate=instance.goal.predicate,
                reference_ref=instance.goal.reference_ref,
                desired_value=instance.goal.desired_value,
            )
        ),
    )


def _validate_execution_matches_program(
    program: CapabilityProgramV1,
    state: CapabilityExecutionStateV1,
) -> None:
    if state.program_id != program.program_id:
        raise ValueError("execution state belongs to a different capability program")
    if tuple(step.instance_id for step in state.steps) != tuple(
        step.instance_id for step in program.steps
    ):
        raise ValueError("execution steps do not match the capability program")


def active_capability_parts_v1(
    program: CapabilityProgramV1,
    state: CapabilityExecutionStateV1,
) -> tuple[
    int,
    CapabilityInstanceV1,
    CapabilityStepStateV1,
    CapabilityTemplateV1,
]:
    _validate_execution_matches_program(program, state)
    if state.active_step_index is None:
        raise ValueError("capability program has no active step")
    step_index = state.active_step_index
    instance = program.steps[step_index]
    step_state = state.steps[step_index]
    template = CAPABILITY_TEMPLATES_V1[instance.kind]
    if step_state.phase_index >= len(template.phases):
        raise ValueError("active phase index exceeds its capability template")
    return step_index, instance, step_state, template
