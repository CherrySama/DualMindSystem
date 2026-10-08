from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from jev4mujoco.perception.surface_geometry import polygon_clearance_v1
from jev4mujoco.contracts.visual_scene import (
    EntityV1,
    EvidenceStatus,
    Polygon2,
    Visibility,
    VisualPredicate,
    VisualRelationV1,
)


@dataclass(frozen=True, slots=True)
class VisualRelationConfigV1:
    directional_gap_m: float
    near_maximum_clearance_m: float
    separated_minimum_clearance_m: float
    support_height_tolerance_m: float
    support_minimum_overlap_fraction: float
    volume_height_tolerance_m: float = 0.010

    def __post_init__(self) -> None:
        for name, value in (
            ("directional_gap_m", self.directional_gap_m),
            ("near_maximum_clearance_m", self.near_maximum_clearance_m),
            ("separated_minimum_clearance_m", self.separated_minimum_clearance_m),
            ("support_height_tolerance_m", self.support_height_tolerance_m),
            ("volume_height_tolerance_m", self.volume_height_tolerance_m),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            self.near_maximum_clearance_m
            >= self.separated_minimum_clearance_m
        ):
            raise ValueError("near and separated thresholds require a dead band")
        if (
            not isfinite(self.support_minimum_overlap_fraction)
            or not 0.0 < self.support_minimum_overlap_fraction <= 1.0
        ):
            raise ValueError(
                "support_minimum_overlap_fraction must be in the interval (0, 1]"
            )


def _bounds(footprint: Polygon2) -> tuple[float, float, float, float]:
    xs = tuple(point[0] for point in footprint)
    ys = tuple(point[1] for point in footprint)
    return min(xs), max(xs), min(ys), max(ys)


def _xy_overlap_fraction(subject: Polygon2, reference: Polygon2) -> float:
    subject_min_x, subject_max_x, subject_min_y, subject_max_y = _bounds(subject)
    reference_min_x, reference_max_x, reference_min_y, reference_max_y = _bounds(
        reference
    )
    overlap_x = max(
        0.0,
        min(subject_max_x, reference_max_x) - max(subject_min_x, reference_min_x),
    )
    overlap_y = max(
        0.0,
        min(subject_max_y, reference_max_y) - max(subject_min_y, reference_min_y),
    )
    subject_area = (subject_max_x - subject_min_x) * (
        subject_max_y - subject_min_y
    )
    return 0.0 if subject_area <= 0.0 else overlap_x * overlap_y / subject_area


def derive_entity_relations_v1(
    entities: tuple[EntityV1, ...],
    config: VisualRelationConfigV1,
) -> tuple[VisualRelationV1, ...]:
    relations: list[VisualRelationV1] = []
    for subject in entities:
        if subject.visibility is not Visibility.CLEAR:
            continue
        subject_footprint = subject.geometry.footprint_xy_robot_base_m.value
        subject_centroid = subject.geometry.centroid_robot_base_m.value
        subject_extent = subject.geometry.extent_m.value
        if (
            subject_footprint is None
            or subject_centroid is None
            or subject_extent is None
        ):
            continue
        subject_min_x, subject_max_x, subject_min_y, subject_max_y = _bounds(
            subject_footprint
        )
        subject_bottom = subject_centroid[2] - 0.5 * subject_extent[2]
        for reference in entities:
            if reference.track_id == subject.track_id:
                continue
            if reference.visibility is not Visibility.CLEAR:
                continue
            reference_footprint = reference.geometry.footprint_xy_robot_base_m.value
            reference_top = reference.geometry.top_center_robot_base_m.value
            if reference_footprint is None or reference_top is None:
                continue
            (
                reference_min_x,
                reference_max_x,
                reference_min_y,
                reference_max_y,
            ) = _bounds(reference_footprint)
            clearance = polygon_clearance_v1(
                subject_footprint,
                reference_footprint,
            )
            overlap_fraction = _xy_overlap_fraction(
                subject_footprint,
                reference_footprint,
            )
            values = (
                (
                    VisualPredicate.LEFT_OF,
                    subject_min_y
                    >= reference_max_y + config.directional_gap_m,
                ),
                (
                    VisualPredicate.RIGHT_OF,
                    subject_max_y
                    <= reference_min_y - config.directional_gap_m,
                ),
                (
                    VisualPredicate.IN_FRONT_OF,
                    subject_min_x
                    >= reference_max_x + config.directional_gap_m,
                ),
                (
                    VisualPredicate.NEAR,
                    clearance <= config.near_maximum_clearance_m,
                ),
                (
                    VisualPredicate.SEPARATED,
                    clearance >= config.separated_minimum_clearance_m,
                ),
                (
                    VisualPredicate.ON_SURFACE,
                    abs(subject_bottom - reference_top[2])
                    <= config.support_height_tolerance_m
                    and overlap_fraction
                    >= config.support_minimum_overlap_fraction,
                ),
            )
            relations.extend(
                VisualRelationV1(
                    predicate=predicate,
                    subject_ref=subject.track_id,
                    reference_ref=reference.track_id,
                    value=value,
                    status=EvidenceStatus.DERIVED,
                    confidence=1.0,
                )
                for predicate, value in values
            )
    return tuple(relations)
