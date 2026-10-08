from __future__ import annotations

from dataclasses import dataclass
from math import radians
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Scene binding and control parameters used by the atomic executor."""

    scene_path: Path
    physics_dt_s: float
    cartesian_step_m: float
    action_timeout_s: float
    rotation_step_rad: float
    robot_base_body: str
    tcp_site: str
    bowl_body: str
    target_body: str | None
    gripper_open_joint_m: float
    gripper_closed_joint_m: float
    gripper_position_kp: float
    gripper_effort_limit_n: float
    translation_error_m: float
    rotation_error_rad: float
    rotation_tcp_drift_m: float
    gripper_error_m: float
    keyframe_name: str
    dls_damping: float
    orientation_weight_m_per_rad: float
    maximum_joint_update_rad: float
    hold_duration_s: float
    free_space_convergence_fraction: float
    held_object_convergence_fraction: float


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _fraction(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0 or result > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {result}")
    return result


def load_stage_config(path: str | Path) -> StageConfig:
    """Load the current executor configuration without Stage 1 task parameters."""
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Executor configuration must contain a YAML mapping")
    if _required(raw, "schema") != "ExecutorConfigV1":
        raise ValueError("Expected ExecutorConfigV1 configuration")

    repository_root = config_path.parent.parent
    robot = _required(raw, "robot")
    frames = _required(raw, "frames")
    acceptance = _required(raw, "acceptance")
    executor = _required(raw, "executor")

    action_mode = str(_required(raw, "action_mode"))
    translation_frame = str(_required(raw, "translation_frame"))
    rotation_joint = str(_required(raw, "rotation_joint"))
    if action_mode != "atomic":
        raise ValueError(f"Only atomic action mode is supported, got {action_mode!r}")
    if translation_frame != "robot_base":
        raise ValueError(f"Only robot_base translation is supported, got {translation_frame!r}")
    if rotation_joint != "joint6":
        raise ValueError(f"Atomic rotation must use joint6, got {rotation_joint!r}")

    scene_path = repository_root / str(_required(raw, "scene_path"))
    if not scene_path.is_file():
        raise ValueError(f"Scene does not exist: {scene_path}")

    return StageConfig(
        scene_path=scene_path,
        physics_dt_s=_positive(_required(raw, "physics_dt_s"), "physics_dt_s"),
        cartesian_step_m=_positive(_required(raw, "cartesian_step_m"), "cartesian_step_m"),
        action_timeout_s=_positive(_required(raw, "action_timeout_s"), "action_timeout_s"),
        rotation_step_rad=radians(_positive(_required(raw, "rotation_step_deg"), "rotation_step_deg")),
        robot_base_body=str(_required(frames, "robot_base_body")),
        tcp_site=str(_required(frames, "tcp_site")),
        bowl_body=str(_required(raw, "object_body")),
        target_body=None if raw.get("target_body") is None else str(raw["target_body"]),
        gripper_open_joint_m=float(_required(robot, "gripper_open_joint_m")),
        gripper_closed_joint_m=float(_required(robot, "gripper_closed_joint_m")),
        gripper_position_kp=_positive(
            _required(robot, "gripper_position_kp"), "gripper_position_kp"
        ),
        gripper_effort_limit_n=_positive(
            _required(robot, "gripper_effort_limit_n"), "gripper_effort_limit_n"
        ),
        translation_error_m=_positive(_required(acceptance, "translation_error_m"), "translation_error_m"),
        rotation_error_rad=radians(_positive(_required(acceptance, "rotation_error_deg"), "rotation_error_deg")),
        rotation_tcp_drift_m=_positive(_required(acceptance, "rotation_tcp_drift_m"), "rotation_tcp_drift_m"),
        gripper_error_m=_positive(_required(acceptance, "gripper_error_m"), "gripper_error_m"),
        keyframe_name=str(_required(executor, "keyframe_name")),
        dls_damping=_positive(_required(executor, "dls_damping"), "dls_damping"),
        orientation_weight_m_per_rad=_positive(
            _required(executor, "orientation_weight_m_per_rad"),
            "orientation_weight_m_per_rad",
        ),
        maximum_joint_update_rad=_positive(
            _required(executor, "maximum_joint_update_rad"), "maximum_joint_update_rad"
        ),
        hold_duration_s=_positive(_required(executor, "hold_duration_s"), "hold_duration_s"),
        free_space_convergence_fraction=_fraction(
            _required(executor, "free_space_convergence_fraction"),
            "free_space_convergence_fraction",
        ),
        held_object_convergence_fraction=_fraction(
            _required(executor, "held_object_convergence_fraction"),
            "held_object_convergence_fraction",
        ),
    )
