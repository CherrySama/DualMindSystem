from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from jev4mujoco.contracts.actions import AtomicAction
from jev4mujoco.contracts.local_intent import LocalIntentChoiceV1
from jev4mujoco.contracts.state_snapshot import StateSnapshotV2


class CapabilityDecisionKind(str, Enum):
    ACTION = "action"
    REQUEST_VERIFICATION = "request_verification"
    BLOCKED = "blocked"
    REPLAN_LOCAL_INTENT = "replan_local_intent"


class RecoveryChoiceV1(str, Enum):
    OBSERVE = "observe"
    RETRY_PICK = "retry_pick"
    RETRY_CAPABILITY = "retry_capability"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class RecoveryDecisionV1:
    choice: RecoveryChoiceV1
    based_on_state_id: int
    based_on_action_epoch: int
    reason: str

    def __post_init__(self) -> None:
        if self.based_on_state_id <= 0 or self.based_on_action_epoch < 0:
            raise ValueError("recovery decision requires a valid state and action epoch")
        if not self.reason.strip():
            raise ValueError("recovery decision requires a reason")


@dataclass(frozen=True, slots=True)
class CapabilityDecisionV1:
    decision_id: int
    kind: CapabilityDecisionKind
    based_on_state_id: int
    based_on_action_epoch: int
    skill_instance_id: str
    action: AtomicAction | None
    reason: str
    local_intent_choice: LocalIntentChoiceV1 | None = None

    def __post_init__(self) -> None:
        if type(self.decision_id) is not int or self.decision_id <= 0:
            raise ValueError("decision_id must be a positive integer")
        if type(self.based_on_state_id) is not int or self.based_on_state_id < 0:
            raise ValueError("based_on_state_id must be a nonnegative integer")
        if type(self.based_on_action_epoch) is not int or self.based_on_action_epoch < 0:
            raise ValueError("based_on_action_epoch must be a nonnegative integer")
        if not self.skill_instance_id.strip():
            raise ValueError("skill_instance_id must be non-empty")
        if not self.reason.strip():
            raise ValueError("decision reason must be non-empty")
        if self.kind is CapabilityDecisionKind.ACTION:
            if self.action is None:
                raise ValueError("action decision requires an atomic action")
        elif self.action is not None:
            raise ValueError("non-action decision cannot carry an atomic action")
        if self.local_intent_choice is not None:
            choice = self.local_intent_choice
            if (choice.based_on_state_id != self.based_on_state_id
                or choice.based_on_action_epoch != self.based_on_action_epoch
                or choice.skill_instance_id != self.skill_instance_id):
                raise ValueError("local intent choice must bind the same decision snapshot")


class CapabilityPolicyV1(Protocol):
    """JEV 原子决策与 Runtime 之间的策略契约。"""

    def decide(self, snapshot: StateSnapshotV2) -> CapabilityDecisionV1:
        """Propose one action or one Runtime-owned verification request."""
