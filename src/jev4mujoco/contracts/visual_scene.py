from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Any, Generic, TypeVar


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]
Polygon2 = tuple[Vec2, ...]

T = TypeVar("T")


class EvidenceStatus(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    UNOBSERVABLE = "unobservable"


class EntityType(str, Enum):
    MOVABLE_OBJECT = "movable_object"
    CONTAINER = "container"
    BUTTON = "button"
    OBSTACLE = "obstacle"


class RegionType(str, Enum):
    REGION_2D = "region_2d"
    SURFACE = "surface"
    VOLUME_3D = "volume_3d"
    POINT = "point"


class Visibility(str, Enum):
    CLEAR = "clear"
    PARTIAL = "partial"
    NOT_DETECTED = "not_detected"


class ObservationScopeMode(str, Enum):
    FULL_WORKSPACE = "full_workspace"
    ATTENTION_SUBSET = "attention_subset"


class RegionSourceKind(str, Enum):
    CALIBRATED = "calibrated"
    MEASURED = "measured"
    DERIVED = "derived"


class VisualPredicate(str, Enum):
    IN_REGION = "in_region"
    CLEAR_OF_REGION = "clear_of_region"
    INSIDE = "inside"
    ON_SURFACE = "on_surface"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    NEAR = "near"
    SEPARATED = "separated"
    PRESSED = "pressed"


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_confidence(value: float, name: str = "confidence") -> None:
    if not isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")


def _require_vector(value: tuple[float, ...], size: int, name: str) -> None:
    if len(value) != size or not all(isfinite(component) for component in value):
        raise ValueError(f"{name} must contain {size} finite values")


def _require_polygon(value: Polygon2, name: str) -> None:
    if len(value) < 3:
        raise ValueError(f"{name} must contain at least three points")
    for index, point in enumerate(value):
        _require_vector(point, 2, f"{name}[{index}]")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class EvidenceValueV1(Generic[T]):
    value: T | None
    status: EvidenceStatus
    confidence: float

    def __post_init__(self) -> None:
        _require_confidence(self.confidence)
        if self.status is EvidenceStatus.UNOBSERVABLE:
            if self.value is not None or self.confidence != 0.0:
                raise ValueError(
                    "unobservable evidence must have value=None and confidence=0"
                )
        elif self.value is None:
            raise ValueError("measured or derived evidence must contain a value")


@dataclass(frozen=True, slots=True)
class PoseV1:
    position_robot_base_m: Vec3
    orientation_robot_base_wxyz: QuatWxyz

    def __post_init__(self) -> None:
        _require_vector(self.position_robot_base_m, 3, "position_robot_base_m")
        _require_vector(
            self.orientation_robot_base_wxyz,
            4,
            "orientation_robot_base_wxyz",
        )


@dataclass(frozen=True, slots=True)
class EntitySemanticV1:
    category: EvidenceValueV1[str]
    color: EvidenceValueV1[str]

    def __post_init__(self) -> None:
        for name, evidence in (("category", self.category), ("color", self.color)):
            if evidence.value is not None:
                _require_identifier(evidence.value, f"semantic.{name}.value")


@dataclass(frozen=True, slots=True)
class EntityGeometryV1:
    centroid_robot_base_m: EvidenceValueV1[Vec3]
    top_center_robot_base_m: EvidenceValueV1[Vec3]
    extent_m: EvidenceValueV1[Vec3]
    footprint_xy_robot_base_m: EvidenceValueV1[Polygon2]
    surface_normal_robot_base: EvidenceValueV1[Vec3]
    pose_6d_robot_base: EvidenceValueV1[PoseV1]

    def __post_init__(self) -> None:
        for name, evidence in (
            ("centroid_robot_base_m", self.centroid_robot_base_m),
            ("top_center_robot_base_m", self.top_center_robot_base_m),
            ("extent_m", self.extent_m),
            ("surface_normal_robot_base", self.surface_normal_robot_base),
        ):
            if evidence.value is not None:
                _require_vector(evidence.value, 3, name)
        if self.extent_m.value is not None and any(
            component <= 0.0 for component in self.extent_m.value
        ):
            raise ValueError("extent_m components must be positive")
        if self.footprint_xy_robot_base_m.value is not None:
            _require_polygon(
                self.footprint_xy_robot_base_m.value,
                "footprint_xy_robot_base_m",
            )


@dataclass(frozen=True, slots=True)
class EntityMeasurementV1:
    method: str
    depth_valid_fraction: float

    def __post_init__(self) -> None:
        _require_identifier(self.method, "measurement.method")
        _require_confidence(
            self.depth_valid_fraction,
            "measurement.depth_valid_fraction",
        )


@dataclass(frozen=True, slots=True)
class EntityV1:
    track_id: str
    entity_type: EntityType
    semantic: EntitySemanticV1
    visibility: Visibility
    geometry: EntityGeometryV1
    support_region_ref: str | None
    measurement: EntityMeasurementV1

    def __post_init__(self) -> None:
        _require_identifier(self.track_id, "track_id")
        if self.support_region_ref is not None:
            _require_identifier(self.support_region_ref, "support_region_ref")


@dataclass(frozen=True, slots=True)
class RegionGeometryV1:
    boundary_xy_robot_base_m: EvidenceValueV1[Polygon2] | None = None
    surface_height_robot_base_m: EvidenceValueV1[float] | None = None
    surface_normal_robot_base: EvidenceValueV1[Vec3] | None = None
    minimum_z_robot_base_m: EvidenceValueV1[float] | None = None
    maximum_z_robot_base_m: EvidenceValueV1[float] | None = None
    point_robot_base_m: EvidenceValueV1[Vec3] | None = None

    def __post_init__(self) -> None:
        if (
            self.boundary_xy_robot_base_m is not None
            and self.boundary_xy_robot_base_m.value is not None
        ):
            _require_polygon(
                self.boundary_xy_robot_base_m.value,
                "boundary_xy_robot_base_m",
            )
        if (
            self.surface_normal_robot_base is not None
            and self.surface_normal_robot_base.value is not None
        ):
            _require_vector(
                self.surface_normal_robot_base.value,
                3,
                "surface_normal_robot_base",
            )
        if self.point_robot_base_m is not None and self.point_robot_base_m.value is not None:
            _require_vector(self.point_robot_base_m.value, 3, "point_robot_base_m")
        if (
            self.minimum_z_robot_base_m is not None
            and self.maximum_z_robot_base_m is not None
            and self.minimum_z_robot_base_m.value is not None
            and self.maximum_z_robot_base_m.value is not None
            and self.minimum_z_robot_base_m.value
            >= self.maximum_z_robot_base_m.value
        ):
            raise ValueError("minimum_z_robot_base_m must be below maximum_z_robot_base_m")


@dataclass(frozen=True, slots=True)
class RegionSourceV1:
    kind: RegionSourceKind
    confidence: float

    def __post_init__(self) -> None:
        _require_confidence(self.confidence, "region.source.confidence")


@dataclass(frozen=True, slots=True)
class RegionV1:
    region_id: str
    region_type: RegionType
    semantic_label: str
    owner_entity_ref: str | None
    geometry: RegionGeometryV1
    source: RegionSourceV1

    def __post_init__(self) -> None:
        _require_identifier(self.region_id, "region_id")
        _require_identifier(self.semantic_label, "semantic_label")
        if self.owner_entity_ref is not None:
            _require_identifier(self.owner_entity_ref, "owner_entity_ref")

        if self.region_type in (RegionType.REGION_2D, RegionType.SURFACE):
            required = (
                self.geometry.boundary_xy_robot_base_m,
                self.geometry.surface_height_robot_base_m,
                self.geometry.surface_normal_robot_base,
            )
            if any(item is None or item.value is None for item in required):
                raise ValueError(
                    f"{self.region_type.value} requires boundary, surface height and normal"
                )
        elif self.region_type is RegionType.VOLUME_3D:
            required = (
                self.geometry.boundary_xy_robot_base_m,
                self.geometry.minimum_z_robot_base_m,
                self.geometry.maximum_z_robot_base_m,
            )
            if any(item is None or item.value is None for item in required):
                raise ValueError("volume_3d requires boundary and minimum/maximum Z")
        elif self.region_type is RegionType.POINT:
            if (
                self.geometry.point_robot_base_m is None
                or self.geometry.point_robot_base_m.value is None
                or self.geometry.surface_normal_robot_base is None
                or self.geometry.surface_normal_robot_base.value is None
            ):
                raise ValueError("point region requires position and surface normal")


@dataclass(frozen=True, slots=True)
class VisualRelationV1:
    predicate: VisualPredicate
    subject_ref: str
    reference_ref: str
    value: bool
    status: EvidenceStatus
    confidence: float

    def __post_init__(self) -> None:
        _require_identifier(self.subject_ref, "relation.subject_ref")
        _require_identifier(self.reference_ref, "relation.reference_ref")
        if self.status is EvidenceStatus.UNOBSERVABLE:
            raise ValueError("published visual relations must be measured or derived")
        _require_confidence(self.confidence, "relation.confidence")


@dataclass(frozen=True, slots=True)
class SensorV1:
    sensor_id: str
    type: str
    mounting: str
    optical_frame: str
    calibration_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("sensor_id", self.sensor_id),
            ("optical_frame", self.optical_frame),
            ("calibration_id", self.calibration_id),
        ):
            _require_identifier(value, f"sensor.{name}")
        if self.type != "rgbd":
            raise ValueError("VisualSceneV1 requires an RGB-D sensor")
        if self.mounting != "fixed":
            raise ValueError("VisualSceneV1 V1 requires a fixed RGB-D camera")


@dataclass(frozen=True, slots=True)
class ObservationScopeV1:
    mode: ObservationScopeMode
    requested_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode is ObservationScopeMode.FULL_WORKSPACE and self.requested_refs:
            raise ValueError("full_workspace scope cannot contain requested_refs")
        if self.mode is ObservationScopeMode.ATTENTION_SUBSET and not self.requested_refs:
            raise ValueError("attention_subset scope requires requested_refs")
        for reference in self.requested_refs:
            _require_identifier(reference, "observation_scope.requested_refs item")


@dataclass(frozen=True, slots=True)
class VisualDataRefsV1:
    rgb: str | None = None
    depth: str | None = None
    instance_masks: str | None = None
    point_cloud: str | None = None


@dataclass(frozen=True, slots=True)
class VisualUnitsV1:
    length: str = "m"

    def __post_init__(self) -> None:
        if self.length != "m":
            raise ValueError("VisualSceneV1 length unit must be metres")


@dataclass(frozen=True, slots=True)
class VisualSceneV1:
    scene_id: str
    observation_id: int
    capture_timestamp_s: float
    publish_timestamp_s: float
    sensor: SensorV1
    observation_scope: ObservationScopeV1
    planning_ready: bool
    entities: tuple[EntityV1, ...]
    regions: tuple[RegionV1, ...]
    relations: tuple[VisualRelationV1, ...]
    data_refs: VisualDataRefsV1
    schema: str = "VisualSceneV1"
    schema_version: int = 1
    coordinate_frame: str = "robot_base"
    units: VisualUnitsV1 = VisualUnitsV1()

    def __post_init__(self) -> None:
        if self.schema != "VisualSceneV1" or self.schema_version != 1:
            raise ValueError("VisualSceneV1 schema identity is fixed")
        _require_identifier(self.scene_id, "scene_id")
        if type(self.observation_id) is not int or self.observation_id <= 0:
            raise ValueError("observation_id must be a positive integer")
        if not isfinite(self.capture_timestamp_s) or self.capture_timestamp_s < 0.0:
            raise ValueError("capture_timestamp_s must be finite and nonnegative")
        if (
            not isfinite(self.publish_timestamp_s)
            or self.publish_timestamp_s < self.capture_timestamp_s
        ):
            raise ValueError("publish_timestamp_s must not precede capture_timestamp_s")
        if self.coordinate_frame != "robot_base":
            raise ValueError("VisualSceneV1 uses robot_base coordinates")
        if self.planning_ready and self.observation_scope.mode is not ObservationScopeMode.FULL_WORKSPACE:
            raise ValueError("planning_ready requires a full_workspace observation")

        entity_ids = [entity.track_id for entity in self.entities]
        region_ids = [region.region_id for region in self.regions]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity track_ids must be unique")
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("region_ids must be unique")
        if set(entity_ids) & set(region_ids):
            raise ValueError("entity and region identifiers must not collide")

        entity_id_set = set(entity_ids)
        region_id_set = set(region_ids)
        all_ids = entity_id_set | region_id_set
        for entity in self.entities:
            if (
                entity.support_region_ref is not None
                and entity.support_region_ref not in region_id_set
            ):
                raise ValueError(
                    f"entity {entity.track_id} references missing support region "
                    f"{entity.support_region_ref}"
                )
        for region in self.regions:
            if (
                region.owner_entity_ref is not None
                and region.owner_entity_ref not in entity_id_set
            ):
                raise ValueError(
                    f"region {region.region_id} references missing owner entity "
                    f"{region.owner_entity_ref}"
                )
        for relation in self.relations:
            if relation.subject_ref not in all_ids or relation.reference_ref not in all_ids:
                raise ValueError("visual relation references an unknown entity or region")

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)
