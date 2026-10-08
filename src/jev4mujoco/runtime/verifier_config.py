"""具名验证参数、参数校验及 YAML 加载。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping
import yaml
from jev4mujoco.perception.relations import VisualRelationConfigV1

@dataclass(frozen=True, slots=True)
class RuntimeVerifierConfigV1:
    max_visual_age_s: float
    visual_relation_config: VisualRelationConfigV1
    gripper_opening_m: float
    gripper_tolerance_m: float
    grasp_xy_tolerance_m: float
    grasp_z_tolerance_m: float
    transport_minimum_bottom_clearance_m: float
    release_maximum_bottom_clearance_m: float
    placement_obstacle_clearance_m: float
    placement_reference_maximum_displacement_m: float
    placement_required_consecutive_observations: int
    placement_minimum_observation_interval_s: float
    placement_maximum_centroid_shift_m: float
    arrangement_row_cross_axis_tolerance_m: float
    arrangement_minimum_pairwise_clearance_m: float
    press_point_xy_tolerance_m: float
    press_minimum_travel_m: float
    press_visual_confirmation_margin_m: float
    surface_push_closed_gripper_position_m: float
    surface_push_gripper_position_tolerance_m: float
    surface_push_precontact_clearance_m: float
    surface_push_precontact_xy_tolerance_m: float
    surface_push_approach_lateral_tolerance_m: float
    surface_push_tcp_height_above_surface_m: float
    surface_push_tcp_height_tolerance_m: float
    surface_push_reposition_height_above_surface_m: float
    surface_push_clear_of_region_margin_m: float
    surface_push_required_consecutive_observations: int
    surface_push_minimum_observation_interval_s: float
    surface_push_maximum_centroid_shift_m: float
    minimum_pick_xy_progress_m: float
    maximum_consecutive_rejections: int
    maximum_consecutive_nonprogress_actions: int
    maximum_pending_verification_s: float

    def __post_init__(self) -> None:
        positive = (
            ("max_visual_age_s", self.max_visual_age_s),
            ("maximum_pending_verification_s", self.maximum_pending_verification_s),
            ("gripper_tolerance_m", self.gripper_tolerance_m),
            ("grasp_xy_tolerance_m", self.grasp_xy_tolerance_m),
            ("grasp_z_tolerance_m", self.grasp_z_tolerance_m),
            (
                "transport_minimum_bottom_clearance_m",
                self.transport_minimum_bottom_clearance_m,
            ),
            (
                "release_maximum_bottom_clearance_m",
                self.release_maximum_bottom_clearance_m,
            ),
            (
                "placement_obstacle_clearance_m",
                self.placement_obstacle_clearance_m,
            ),
            (
                "placement_reference_maximum_displacement_m",
                self.placement_reference_maximum_displacement_m,
            ),
            (
                "placement_minimum_observation_interval_s",
                self.placement_minimum_observation_interval_s,
            ),
            (
                "placement_maximum_centroid_shift_m",
                self.placement_maximum_centroid_shift_m,
            ),
            (
                "arrangement_row_cross_axis_tolerance_m",
                self.arrangement_row_cross_axis_tolerance_m,
            ),
            (
                "arrangement_minimum_pairwise_clearance_m",
                self.arrangement_minimum_pairwise_clearance_m,
            ),
            ("press_point_xy_tolerance_m", self.press_point_xy_tolerance_m),
            ("press_minimum_travel_m", self.press_minimum_travel_m),
            (
                "press_visual_confirmation_margin_m",
                self.press_visual_confirmation_margin_m,
            ),
            (
                "surface_push_gripper_position_tolerance_m",
                self.surface_push_gripper_position_tolerance_m,
            ),
            (
                "surface_push_precontact_clearance_m",
                self.surface_push_precontact_clearance_m,
            ),
            (
                "surface_push_precontact_xy_tolerance_m",
                self.surface_push_precontact_xy_tolerance_m,
            ),
            (
                "surface_push_approach_lateral_tolerance_m",
                self.surface_push_approach_lateral_tolerance_m,
            ),
            (
                "surface_push_tcp_height_above_surface_m",
                self.surface_push_tcp_height_above_surface_m,
            ),
            (
                "surface_push_tcp_height_tolerance_m",
                self.surface_push_tcp_height_tolerance_m,
            ),
            (
                "surface_push_reposition_height_above_surface_m",
                self.surface_push_reposition_height_above_surface_m,
            ),
            (
                "surface_push_minimum_observation_interval_s",
                self.surface_push_minimum_observation_interval_s,
            ),
            (
                "surface_push_maximum_centroid_shift_m",
                self.surface_push_maximum_centroid_shift_m,
            ),
        )
        for name, value in positive:
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("gripper_opening_m", self.gripper_opening_m),
            (
                "surface_push_closed_gripper_position_m",
                self.surface_push_closed_gripper_position_m,
            ),
            (
                "surface_push_clear_of_region_margin_m",
                self.surface_push_clear_of_region_margin_m,
            ),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not isfinite(self.minimum_pick_xy_progress_m)
            or self.minimum_pick_xy_progress_m < 0.0
        ):
            raise ValueError("minimum_pick_xy_progress_m must be finite and nonnegative")
        for name in (
            "maximum_consecutive_rejections",
            "maximum_consecutive_nonprogress_actions",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.placement_required_consecutive_observations) is not int
            or self.placement_required_consecutive_observations <= 0
        ):
            raise ValueError(
                "placement_required_consecutive_observations must be positive"
            )
        if (
            type(self.surface_push_required_consecutive_observations) is not int
            or self.surface_push_required_consecutive_observations <= 0
        ):
            raise ValueError(
                "surface_push_required_consecutive_observations must be positive"
            )


def load_runtime_verifier_config_v1(path: str | Path) -> RuntimeVerifierConfigV1:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict) or raw.get("schema") != "RuntimeVerifierConfigV1":
        raise ValueError("runtime verifier schema must be RuntimeVerifierConfigV1")

    def required(mapping: Mapping[str, Any], key: str) -> Any:
        if key not in mapping:
            raise ValueError(f"missing runtime verifier configuration key: {key}")
        return mapping[key]

    visual = required(raw, "visual")
    visual_relations = required(raw, "visual_relations")
    gripper = required(raw, "gripper")
    grasp = required(raw, "grasp")
    action_review = required(raw, "action_review")
    transport = required(raw, "transport")
    release = required(raw, "release")
    placement = required(raw, "placement")
    arrangement = required(raw, "arrangement")
    press = required(raw, "press")
    surface_push = required(raw, "surface_push")
    return RuntimeVerifierConfigV1(
        max_visual_age_s=float(required(visual, "maximum_age_s")),
        visual_relation_config=VisualRelationConfigV1(
            directional_gap_m=float(
                required(visual_relations, "directional_gap_m")
            ),
            near_maximum_clearance_m=float(
                required(visual_relations, "near_maximum_clearance_m")
            ),
            separated_minimum_clearance_m=float(
                required(visual_relations, "separated_minimum_clearance_m")
            ),
            support_height_tolerance_m=float(
                required(visual_relations, "support_height_tolerance_m")
            ),
            support_minimum_overlap_fraction=float(
                required(visual_relations, "support_minimum_overlap_fraction")
            ),
            volume_height_tolerance_m=float(visual_relations.get("volume_height_tolerance_m", 0.010)),
        ),
        gripper_opening_m=float(required(gripper, "open_position_m")),
        gripper_tolerance_m=float(required(gripper, "position_tolerance_m")),
        grasp_xy_tolerance_m=float(required(grasp, "xy_tolerance_m")),
        grasp_z_tolerance_m=float(required(grasp, "z_tolerance_m")),
        minimum_pick_xy_progress_m=float(
            required(action_review, "minimum_pick_xy_progress_m")
        ),
        maximum_consecutive_rejections=int(
            required(action_review, "maximum_consecutive_rejections")
        ),
        maximum_consecutive_nonprogress_actions=int(
            required(action_review, "maximum_consecutive_nonprogress_actions")
        ),
        maximum_pending_verification_s=float(
            required(required(raw, "verification"), "maximum_pending_seconds")
        ),
        transport_minimum_bottom_clearance_m=float(
            required(transport, "minimum_bottom_clearance_m")
        ),
        release_maximum_bottom_clearance_m=float(
            required(release, "maximum_bottom_clearance_m")
        ),
        placement_obstacle_clearance_m=float(
            required(placement, "obstacle_clearance_m")
        ),
        placement_reference_maximum_displacement_m=float(
            required(placement, "reference_maximum_displacement_m")
        ),
        placement_required_consecutive_observations=int(
            required(placement, "required_consecutive_observations")
        ),
        placement_minimum_observation_interval_s=float(
            required(placement, "minimum_observation_interval_s")
        ),
        placement_maximum_centroid_shift_m=float(
            required(placement, "maximum_centroid_shift_m")
        ),
        arrangement_row_cross_axis_tolerance_m=float(
            required(arrangement, "row_cross_axis_tolerance_m")
        ),
        arrangement_minimum_pairwise_clearance_m=float(
            required(arrangement, "minimum_pairwise_clearance_m")
        ),
        press_point_xy_tolerance_m=float(required(press, "point_xy_tolerance_m")),
        press_minimum_travel_m=float(required(press, "minimum_travel_m")),
        press_visual_confirmation_margin_m=float(
            required(press, "visual_confirmation_margin_m")
        ),
        surface_push_closed_gripper_position_m=float(
            required(surface_push, "closed_gripper_position_m")
        ),
        surface_push_gripper_position_tolerance_m=float(
            required(surface_push, "gripper_position_tolerance_m")
        ),
        surface_push_precontact_clearance_m=float(
            required(surface_push, "precontact_clearance_m")
        ),
        surface_push_precontact_xy_tolerance_m=float(
            required(surface_push, "precontact_xy_tolerance_m")
        ),
        surface_push_approach_lateral_tolerance_m=float(
            required(surface_push, "approach_lateral_tolerance_m")
        ),
        surface_push_tcp_height_above_surface_m=float(
            required(surface_push, "tcp_height_above_surface_m")
        ),
        surface_push_tcp_height_tolerance_m=float(
            required(surface_push, "tcp_height_tolerance_m")
        ),
        surface_push_reposition_height_above_surface_m=float(
            required(surface_push, "reposition_height_above_surface_m")
        ),
        surface_push_clear_of_region_margin_m=float(
            required(surface_push, "clear_of_region_margin_m")
        ),
        surface_push_required_consecutive_observations=int(
            required(surface_push, "required_consecutive_observations")
        ),
        surface_push_minimum_observation_interval_s=float(
            required(surface_push, "minimum_observation_interval_s")
        ),
        surface_push_maximum_centroid_shift_m=float(
            required(surface_push, "maximum_centroid_shift_m")
        ),
    )
