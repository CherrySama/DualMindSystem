from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan, sqrt
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import mujoco
import numpy as np
from numpy.typing import NDArray
import yaml

from jev4mujoco.perception.surface_geometry import footprint_clear_of_region_v1
from jev4mujoco.perception.relations import (
    VisualRelationConfigV1,
    derive_entity_relations_v1,
)
from jev4mujoco.contracts.visual_scene import (
    EntityGeometryV1,
    EntityMeasurementV1,
    EntitySemanticV1,
    EntityType,
    EntityV1,
    EvidenceStatus,
    EvidenceValueV1,
    ObservationScopeMode,
    ObservationScopeV1,
    Polygon2,
    RegionGeometryV1,
    RegionSourceKind,
    RegionSourceV1,
    RegionType,
    RegionV1,
    SensorV1,
    Visibility,
    VisualDataRefsV1,
    VisualPredicate,
    VisualRelationV1,
    VisualSceneV1,
)


FloatImage = NDArray[np.float32]
RgbImage = NDArray[np.uint8]
SegmentationImage = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class MujocoRgbdEntitySpecV1:
    """Simulation-only binding between an ideal instance mask and a track."""

    track_id: str
    geom_name: str
    category: str
    color: str
    support_region_ref: str
    entity_type: EntityType = EntityType.MOVABLE_OBJECT

    def __post_init__(self) -> None:
        for name, value in (
            ("track_id", self.track_id),
            ("geom_name", self.geom_name),
            ("category", self.category),
            ("color", self.color),
            ("support_region_ref", self.support_region_ref),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CalibratedSurfaceRegionV1:
    """A fixed workstation region already calibrated in robot-base coordinates."""

    region_id: str
    semantic_label: str
    boundary_xy_robot_base_m: Polygon2 | None
    surface_height_robot_base_m: float | None
    surface_normal_robot_base: tuple[float, float, float] = (0.0, 0.0, 1.0)
    region_type: RegionType = RegionType.REGION_2D
    owner_entity_ref: str | None = None
    minimum_z_robot_base_m: float | None = None
    maximum_z_robot_base_m: float | None = None
    point_robot_base_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.region_type in (RegionType.REGION_2D, RegionType.SURFACE):
            if self.surface_height_robot_base_m is None:
                raise ValueError("surface regions require a surface height")
        elif self.region_type is RegionType.VOLUME_3D:
            if (
                self.owner_entity_ref is None
                or self.minimum_z_robot_base_m is None
                or self.maximum_z_robot_base_m is None
            ):
                raise ValueError("volume regions require owner and minimum/maximum Z")
            if self.minimum_z_robot_base_m >= self.maximum_z_robot_base_m:
                raise ValueError("volume minimum Z must be below maximum Z")
        elif self.region_type is RegionType.POINT:
            if self.owner_entity_ref is None or self.point_robot_base_m is None:
                raise ValueError("point regions require an owner and point")
        else:
            raise ValueError("unsupported calibrated region type")


@dataclass(frozen=True, slots=True)
class FixedRgbdCaptureV1:
    scene: VisualSceneV1
    rgb: RgbImage
    depth_m: FloatImage
    instance_segmentation: SegmentationImage
    entity_points_robot_base_m: Mapping[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class MujocoFixedRgbdConfigV1:
    scene_path: Path
    keyframe_name: str
    camera_name: str
    robot_base_body_name: str
    width: int
    height: int
    sensor_id: str
    calibration_id: str
    entity_specs: tuple[MujocoRgbdEntitySpecV1, ...]
    calibrated_regions: tuple[CalibratedSurfaceRegionV1, ...]

    def make_builder(
        self,
        model: mujoco.MjModel,
        *,
        clear_of_region_margin_m: float = 0.0,
        relation_config: VisualRelationConfigV1 | None = None,
        button_press_minimum_travel_m: float = 0.0,
    ) -> MujocoFixedRgbdVisualSceneBuilderV1:
        return MujocoFixedRgbdVisualSceneBuilderV1(
            model,
            camera_name=self.camera_name,
            robot_base_body_name=self.robot_base_body_name,
            entity_specs=self.entity_specs,
            calibrated_regions=self.calibrated_regions,
            width=self.width,
            height=self.height,
            sensor_id=self.sensor_id,
            calibration_id=self.calibration_id,
            clear_of_region_margin_m=clear_of_region_margin_m,
            relation_config=relation_config,
            button_press_minimum_travel_m=button_press_minimum_travel_m,
        )


class MujocoFixedRgbdVisualSceneBuilderV1:
    """Builds VisualSceneV1 from rendered fixed-camera RGB-D observations.

    MuJoCo geometry IDs provide ideal instance masks in this first simulation
    adapter. Published position and size values are still reconstructed from
    rendered depth; object body poses and free-joint coordinates are never read.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        camera_name: str,
        robot_base_body_name: str,
        entity_specs: tuple[MujocoRgbdEntitySpecV1, ...],
        calibrated_regions: tuple[CalibratedSurfaceRegionV1, ...],
        width: int = 640,
        height: int = 480,
        sensor_id: str = "fixed_rgbd",
        calibration_id: str = "sim_fixed_rgbd_v1",
        clear_of_region_margin_m: float = 0.0,
        relation_config: VisualRelationConfigV1 | None = None,
        button_press_minimum_travel_m: float = 0.0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("image width and height must be positive")
        if not entity_specs:
            raise ValueError("at least one entity spec is required")
        if not calibrated_regions:
            raise ValueError("at least one calibrated region is required")
        if clear_of_region_margin_m < 0.0:
            raise ValueError("clear_of_region_margin_m must be nonnegative")

        self._model = model
        self._camera_name = camera_name
        self._width = width
        self._height = height
        self._sensor_id = sensor_id
        self._calibration_id = calibration_id
        self._clear_of_region_margin_m = clear_of_region_margin_m
        self._relation_config = relation_config
        self._button_press_minimum_travel_m = button_press_minimum_travel_m
        self._entity_specs = entity_specs
        self._calibrated_regions = calibrated_regions
        self._tracked_extent_z_m: dict[str, float] = {}
        self._tracked_min_block_span_m: dict[str, float] = {}

        self._camera_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        self._robot_base_body_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_BODY, robot_base_body_name
        )
        self._geom_ids = {
            spec.track_id: self._require_named_id(
                mujoco.mjtObj.mjOBJ_GEOM, spec.geom_name
            )
            for spec in entity_specs
        }
        if len(self._geom_ids) != len(entity_specs):
            raise ValueError("entity track_ids must be unique")

        region_ids = {region.region_id for region in calibrated_regions}
        if len(region_ids) != len(calibrated_regions):
            raise ValueError("calibrated region IDs must be unique")
        for spec in entity_specs:
            if spec.support_region_ref not in region_ids:
                raise ValueError(
                    f"entity {spec.track_id} references unknown support region "
                    f"{spec.support_region_ref}"
                )

        if (
            int(model.cam_projection[self._camera_id])
            == mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC.value
        ):
            raise ValueError("V1 depth back-projection expects a perspective camera")
        fovy_rad = radians(float(model.cam_fovy[self._camera_id]))
        self._fy = 0.5 * height / tan(0.5 * fovy_rad)
        self._fx = self._fy
        self._cx = 0.5 * (width - 1)
        self._cy = 0.5 * (height - 1)
        self._maximum_depth_m = float(model.vis.map.zfar * model.stat.extent)
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> MujocoFixedRgbdVisualSceneBuilderV1:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def capture(
        self,
        data: mujoco.MjData,
        *,
        scene_id: str,
        observation_id: int,
        capture_timestamp_s: float | None = None,
        publish_timestamp_s: float | None = None,
    ) -> FixedRgbdCaptureV1:
        capture_timestamp = (
            float(data.time)
            if capture_timestamp_s is None
            else float(capture_timestamp_s)
        )
        publish_timestamp = (
            capture_timestamp
            if publish_timestamp_s is None
            else float(publish_timestamp_s)
        )

        self._renderer.update_scene(data, camera=self._camera_name)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        depth_m = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        self._renderer.enable_segmentation_rendering()
        instance_segmentation = self._renderer.render().copy()
        self._renderer.disable_segmentation_rendering()

        camera_position_world = np.asarray(
            data.cam_xpos[self._camera_id], dtype=np.float64
        ).copy()
        camera_rotation_world_from_camera = np.asarray(
            data.cam_xmat[self._camera_id], dtype=np.float64
        ).reshape(3, 3).copy()
        base_position_world = np.asarray(
            data.xpos[self._robot_base_body_id], dtype=np.float64
        ).copy()
        base_rotation_world_from_base = np.asarray(
            data.xmat[self._robot_base_body_id], dtype=np.float64
        ).reshape(3, 3).copy()

        regions = tuple(self._build_region(spec) for spec in self._calibrated_regions)
        region_by_id = {region.region_id: region for region in regions}
        entities: list[EntityV1] = []
        entity_points: dict[str, NDArray[np.float64]] = {}
        for spec in self._entity_specs:
            mask = self._geom_mask(instance_segmentation, self._geom_ids[spec.track_id])
            points = self._back_project_mask_to_robot_base(
                depth_m,
                mask,
                camera_position_world,
                camera_rotation_world_from_camera,
                base_position_world,
                base_rotation_world_from_base,
            )
            entity_points[spec.track_id] = points
            entities.append(
                self._build_entity(
                    spec,
                    mask_pixel_count=int(mask.sum()),
                    points_robot_base_m=points,
                    support_region=region_by_id[spec.support_region_ref],
                    xy_span_uncertainty_m=(0.0 if not len(points) else
                        2 * sqrt(2) * float(np.max(np.linalg.norm(points - np.einsum(
                            "ji,j->i", base_rotation_world_from_base,
                            camera_position_world - base_position_world), axis=1))) / min(self._fx, self._fy)),
                )
            )

        relations = self._derive_relations(
            tuple(entities),
            regions,
            self._clear_of_region_margin_m,
            self._relation_config,
            self._button_press_minimum_travel_m,
        )
        planning_ready = all(
            entity.visibility is Visibility.CLEAR for entity in entities
        )
        data_prefix = f"capture://{scene_id}/{observation_id}"
        scene = VisualSceneV1(
            scene_id=scene_id,
            observation_id=observation_id,
            capture_timestamp_s=capture_timestamp,
            publish_timestamp_s=publish_timestamp,
            sensor=SensorV1(
                sensor_id=self._sensor_id,
                type="rgbd",
                mounting="fixed",
                optical_frame=f"{self._camera_name}_optical",
                calibration_id=self._calibration_id,
            ),
            observation_scope=ObservationScopeV1(
                mode=ObservationScopeMode.FULL_WORKSPACE
            ),
            planning_ready=planning_ready,
            entities=tuple(entities),
            regions=regions,
            relations=relations,
            data_refs=VisualDataRefsV1(
                rgb=f"{data_prefix}/rgb",
                depth=f"{data_prefix}/depth_m",
                instance_masks=f"{data_prefix}/instance_segmentation",
                point_cloud=f"{data_prefix}/entity_points_robot_base_m",
            ),
        )

        for array in (rgb, depth_m, instance_segmentation):
            array.setflags(write=False)
        for points in entity_points.values():
            points.setflags(write=False)
        return FixedRgbdCaptureV1(
            scene=scene,
            rgb=rgb,
            depth_m=depth_m,
            instance_segmentation=instance_segmentation,
            entity_points_robot_base_m=MappingProxyType(entity_points),
        )

    def _require_named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self._model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo {object_type.name} named {name!r} does not exist")
        return object_id

    @staticmethod
    def _geom_mask(segmentation: SegmentationImage, geom_id: int) -> NDArray[np.bool_]:
        return (segmentation[:, :, 0] == geom_id) & (
            segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM.value
        )

    def _back_project_mask_to_robot_base(
        self,
        depth_m: FloatImage,
        mask: NDArray[np.bool_],
        camera_position_world: NDArray[np.float64],
        camera_rotation_world_from_camera: NDArray[np.float64],
        base_position_world: NDArray[np.float64],
        base_rotation_world_from_base: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        depths = depth_m[rows, columns].astype(np.float64)
        valid = (
            np.isfinite(depths)
            & (depths > 0.0)
            & (depths < self._maximum_depth_m)
        )
        rows = rows[valid]
        columns = columns[valid]
        depths = depths[valid]
        if depths.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        points_camera = np.column_stack(
            (
                (columns - self._cx) * depths / self._fx,
                -(rows - self._cy) * depths / self._fy,
                -depths,
            )
        )
        points_world = np.einsum(
            "ij,kj->ik", points_camera, camera_rotation_world_from_camera
        ) + camera_position_world
        return np.einsum(
            "ij,jk->ik",
            points_world - base_position_world,
            base_rotation_world_from_base,
        )

    @staticmethod
    def _build_region(spec: CalibratedSurfaceRegionV1) -> RegionV1:
        measured_boundary = (
            None
            if spec.boundary_xy_robot_base_m is None
            else EvidenceValueV1(
                value=spec.boundary_xy_robot_base_m,
                status=EvidenceStatus.MEASURED,
                confidence=1.0,
            )
        )
        measured_height = (
            None
            if spec.surface_height_robot_base_m is None
            else EvidenceValueV1(
                value=spec.surface_height_robot_base_m,
                status=EvidenceStatus.MEASURED,
                confidence=1.0,
            )
        )
        measured_normal = EvidenceValueV1(
            value=spec.surface_normal_robot_base,
            status=EvidenceStatus.MEASURED,
            confidence=1.0,
        )
        return RegionV1(
            region_id=spec.region_id,
            region_type=spec.region_type,
            semantic_label=spec.semantic_label,
            owner_entity_ref=spec.owner_entity_ref,
            geometry=RegionGeometryV1(
                boundary_xy_robot_base_m=measured_boundary,
                surface_height_robot_base_m=measured_height,
                surface_normal_robot_base=measured_normal,
                minimum_z_robot_base_m=(
                    None
                    if spec.minimum_z_robot_base_m is None
                    else EvidenceValueV1(
                        value=spec.minimum_z_robot_base_m,
                        status=EvidenceStatus.MEASURED,
                        confidence=1.0,
                    )
                ),
                maximum_z_robot_base_m=(
                    None
                    if spec.maximum_z_robot_base_m is None
                    else EvidenceValueV1(
                        value=spec.maximum_z_robot_base_m,
                        status=EvidenceStatus.MEASURED,
                        confidence=1.0,
                    )
                ),
                point_robot_base_m=(
                    None
                    if spec.point_robot_base_m is None
                    else EvidenceValueV1(
                        value=spec.point_robot_base_m,
                        status=EvidenceStatus.MEASURED,
                        confidence=1.0,
                    )
                ),
            ),
            source=RegionSourceV1(kind=RegionSourceKind.CALIBRATED, confidence=1.0),
        )

    def _build_entity(
        self,
        spec: MujocoRgbdEntitySpecV1,
        *,
        mask_pixel_count: int,
        points_robot_base_m: NDArray[np.float64],
        support_region: RegionV1,
        xy_span_uncertainty_m: float = 0.0,
    ) -> EntityV1:
        valid_count = int(points_robot_base_m.shape[0])
        valid_fraction = valid_count / mask_pixel_count if mask_pixel_count else 0.0
        measured_semantic = lambda value: EvidenceValueV1(  # noqa: E731
            value=value, status=EvidenceStatus.MEASURED, confidence=1.0
        )
        unobservable = EvidenceValueV1(
            value=None, status=EvidenceStatus.UNOBSERVABLE, confidence=0.0
        )
        if valid_count < 4:
            geometry = EntityGeometryV1(
                centroid_robot_base_m=unobservable,
                top_center_robot_base_m=unobservable,
                extent_m=unobservable,
                footprint_xy_robot_base_m=unobservable,
                surface_normal_robot_base=unobservable,
                pose_6d_robot_base=unobservable,
            )
            visibility = Visibility.NOT_DETECTED
            current_support_ref = None
        else:
            xy_min = np.min(points_robot_base_m[:, :2], axis=0)
            xy_max = np.max(points_robot_base_m[:, :2], axis=0)
            support_height = support_region.geometry.surface_height_robot_base_m
            assert support_height is not None and support_height.value is not None
            top_z = float(np.percentile(points_robot_base_m[:, 2], 90.0))
            calibrated_support_height = float(support_height.value)
            tracked_height = self._tracked_extent_z_m.get(spec.track_id)
            if tracked_height is None:
                height = top_z - calibrated_support_height
                self._tracked_extent_z_m[spec.track_id] = height
            else:
                height = tracked_height
            extent = (
                float(xy_max[0] - xy_min[0]),
                float(xy_max[1] - xy_min[1]),
                height,
            )
            if min(extent) <= 0.0:
                raise ValueError(
                    f"non-positive RGB-D extent recovered for {spec.track_id}: {extent}"
                )
            center_xy = 0.5 * (xy_min + xy_max)
            center_z = top_z - 0.5 * height
            footprint: Polygon2 = (
                (float(xy_min[0]), float(xy_min[1])),
                (float(xy_max[0]), float(xy_min[1])),
                (float(xy_max[0]), float(xy_max[1])),
                (float(xy_min[0]), float(xy_max[1])),
            )
            support_boundary = support_region.geometry.boundary_xy_robot_base_m
            assert support_boundary is not None and support_boundary.value is not None
            inside_support = all(
                self._point_in_polygon(point, support_boundary.value)
                for point in footprint
            )
            bottom_z = center_z - 0.5 * height
            supported = inside_support and abs(
                bottom_z - calibrated_support_height
            ) <= 0.01
            current_support_ref = spec.support_region_ref if supported else None
            derived = lambda value: EvidenceValueV1(  # noqa: E731
                value=value, status=EvidenceStatus.DERIVED, confidence=1.0
            )
            geometry = EntityGeometryV1(
                centroid_robot_base_m=derived(
                    (
                        float(center_xy[0]),
                        float(center_xy[1]),
                        center_z,
                    )
                ),
                top_center_robot_base_m=derived(
                    (float(center_xy[0]), float(center_xy[1]), top_z)
                ),
                extent_m=derived(extent),
                footprint_xy_robot_base_m=derived(footprint),
                surface_normal_robot_base=derived((0.0, 0.0, 1.0)),
                pose_6d_robot_base=unobservable,
            )
            visibility = Visibility.CLEAR
            if spec.category == "block":
                reference_span = self._tracked_min_block_span_m.get(spec.track_id)
                if reference_span is None:
                    # Conservative first-frame lower bound for the current upright blocks.
                    self._tracked_min_block_span_m[spec.track_id] = min(extent)
                elif min(extent[:2]) + xy_span_uncertainty_m < reference_span:
                    # Visible fragments cannot certify the whole footprint or its centroid.
                    visibility = Visibility.PARTIAL
                    geometry = EntityGeometryV1(unobservable, unobservable, unobservable,
                        unobservable, unobservable, unobservable)
                    current_support_ref = None

        return EntityV1(
            track_id=spec.track_id,
            entity_type=spec.entity_type,
            semantic=EntitySemanticV1(
                category=measured_semantic(spec.category),
                color=measured_semantic(spec.color),
            ),
            visibility=visibility,
            geometry=geometry,
            support_region_ref=current_support_ref,
            measurement=EntityMeasurementV1(
                method="rgbd_instance_mask_depth_backprojection",
                depth_valid_fraction=valid_fraction,
            ),
        )

    @classmethod
    def _derive_relations(
        cls,
        entities: tuple[EntityV1, ...],
        regions: tuple[RegionV1, ...],
        clear_of_region_margin_m: float = 0.0,
        relation_config: VisualRelationConfigV1 | None = None,
        button_press_minimum_travel_m: float = 0.0,
    ) -> tuple[VisualRelationV1, ...]:
        relations: list[VisualRelationV1] = []
        for entity in entities:
            footprint_evidence = entity.geometry.footprint_xy_robot_base_m
            centroid_evidence = entity.geometry.centroid_robot_base_m
            extent_evidence = entity.geometry.extent_m
            if (
                footprint_evidence.value is None
                or centroid_evidence.value is None
                or extent_evidence.value is None
            ):
                continue
            for region in regions:
                if (
                    region.region_type is RegionType.POINT
                    and region.owner_entity_ref == entity.track_id
                ):
                    point = region.geometry.point_robot_base_m
                    top = entity.geometry.top_center_robot_base_m
                    if (
                        point is not None
                        and point.value is not None
                        and top.value is not None
                    ):
                        relations.append(
                            VisualRelationV1(
                                predicate=VisualPredicate.PRESSED,
                                subject_ref=entity.track_id,
                                reference_ref=region.region_id,
                                value=(
                                    top.value[2]
                                    <= point.value[2]
                                    - button_press_minimum_travel_m
                                ),
                                status=EvidenceStatus.DERIVED,
                                confidence=1.0,
                            )
                        )
                    continue
                boundary_evidence = region.geometry.boundary_xy_robot_base_m
                height_evidence = region.geometry.surface_height_robot_base_m
                if boundary_evidence is None or boundary_evidence.value is None:
                    continue
                inside = all(
                    cls._point_in_polygon(point, boundary_evidence.value)
                    for point in footprint_evidence.value
                )
                if region.region_type is RegionType.VOLUME_3D:
                    minimum_z = region.geometry.minimum_z_robot_base_m
                    maximum_z = region.geometry.maximum_z_robot_base_m
                    assert minimum_z is not None and minimum_z.value is not None
                    assert maximum_z is not None and maximum_z.value is not None
                    height_tolerance = (0.010 if relation_config is None else
                                        relation_config.volume_height_tolerance_m)
                    bottom_z = (
                        centroid_evidence.value[2] - 0.5 * extent_evidence.value[2]
                    )
                    top_z = (
                        centroid_evidence.value[2] + 0.5 * extent_evidence.value[2]
                    )
                    relations.append(
                        VisualRelationV1(
                            predicate=VisualPredicate.INSIDE,
                            subject_ref=entity.track_id,
                            reference_ref=region.region_id,
                            value=(
                                inside
                                and bottom_z >= float(minimum_z.value) - height_tolerance
                                and top_z <= float(maximum_z.value) + height_tolerance
                            ),
                            status=EvidenceStatus.DERIVED,
                            confidence=1.0,
                        )
                    )
                    continue
                relations.append(
                    VisualRelationV1(
                        predicate=VisualPredicate.IN_REGION,
                        subject_ref=entity.track_id,
                        reference_ref=region.region_id,
                        value=inside,
                        status=EvidenceStatus.DERIVED,
                        confidence=1.0,
                    )
                )
                relations.append(
                    VisualRelationV1(
                        predicate=VisualPredicate.CLEAR_OF_REGION,
                        subject_ref=entity.track_id,
                        reference_ref=region.region_id,
                        value=footprint_clear_of_region_v1(
                            footprint_evidence.value,
                            boundary_evidence.value,
                            clear_of_region_margin_m,
                        ),
                        status=EvidenceStatus.DERIVED,
                        confidence=1.0,
                    )
                )
                if region.region_id == entity.support_region_ref and height_evidence:
                    bottom_z = (
                        centroid_evidence.value[2] - 0.5 * extent_evidence.value[2]
                    )
                    on_surface = inside and abs(
                        bottom_z - float(height_evidence.value)
                    ) <= 0.01
                    relations.append(
                        VisualRelationV1(
                            predicate=VisualPredicate.ON_SURFACE,
                            subject_ref=entity.track_id,
                            reference_ref=region.region_id,
                            value=on_surface,
                            status=EvidenceStatus.DERIVED,
                            confidence=1.0,
                        )
                    )
        if relation_config is not None:
            relations.extend(derive_entity_relations_v1(entities, relation_config))
        return tuple(relations)

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: Polygon2) -> bool:
        x, y = point
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = previous
            x2, y2 = current
            if (y1 > y) != (y2 > y):
                intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x <= intersection_x:
                    inside = not inside
            previous = current
        return inside


def load_mujoco_fixed_rgbd_config_v1(path: str | Path) -> MujocoFixedRgbdConfigV1:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("fixed RGB-D configuration must be a YAML mapping")
    if raw.get("schema") != "MujocoFixedRgbdConfigV1":
        raise ValueError("fixed RGB-D configuration schema must be MujocoFixedRgbdConfigV1")

    def required(mapping: Mapping[str, Any], key: str) -> Any:
        if key not in mapping:
            raise ValueError(f"missing fixed RGB-D configuration key: {key}")
        return mapping[key]

    camera = required(raw, "camera")
    if not isinstance(camera, dict):
        raise ValueError("camera must be a YAML mapping")
    if required(camera, "type") != "rgbd" or required(camera, "mounting") != "fixed":
        raise ValueError("V1 requires one fixed RGB-D camera")
    width = int(required(camera, "width"))
    height = int(required(camera, "height"))
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")

    entity_items = required(raw, "entities")
    if not isinstance(entity_items, list) or not entity_items:
        raise ValueError("entities must be a non-empty YAML list")
    entity_specs = tuple(
        MujocoRgbdEntitySpecV1(
            track_id=str(required(item, "track_id")),
            geom_name=str(required(item, "segmentation_geom")),
            category=str(required(item, "category")),
            color=str(required(item, "color")),
            support_region_ref=str(required(item, "support_region_ref")),
            entity_type=EntityType(
                str(item.get("entity_type", EntityType.MOVABLE_OBJECT.value))
            ),
        )
        for item in entity_items
    )

    region_items = required(raw, "regions")
    if not isinstance(region_items, list) or not region_items:
        raise ValueError("regions must be a non-empty YAML list")
    calibrated_regions: list[CalibratedSurfaceRegionV1] = []
    for item in region_items:
        if required(item, "coordinate_frame") != "robot_base":
            raise ValueError("all calibrated regions must use robot_base coordinates")
        boundary_items = item.get("boundary_xy_m")
        if boundary_items is not None and not isinstance(boundary_items, list):
            raise ValueError("region boundary_xy_m must be a YAML list")
        boundary: Polygon2 | None = (
            None
            if boundary_items is None
            else tuple((float(point[0]), float(point[1])) for point in boundary_items)
        )
        normal_items = required(item, "surface_normal")
        normal = tuple(float(value) for value in normal_items)
        if len(normal) != 3:
            raise ValueError("region surface_normal must contain three values")
        region_type = RegionType(str(required(item, "region_type")))
        calibrated_regions.append(
            CalibratedSurfaceRegionV1(
                region_id=str(required(item, "region_id")),
                semantic_label=str(required(item, "semantic_label")),
                boundary_xy_robot_base_m=boundary,
                surface_height_robot_base_m=(
                    None
                    if "surface_height_m" not in item
                    else float(item["surface_height_m"])
                ),
                surface_normal_robot_base=normal,  # type: ignore[arg-type]
                region_type=region_type,
                owner_entity_ref=(
                    None
                    if "owner_entity_ref" not in item
                    else str(item["owner_entity_ref"])
                ),
                minimum_z_robot_base_m=(
                    None if "minimum_z_m" not in item else float(item["minimum_z_m"])
                ),
                maximum_z_robot_base_m=(
                    None if "maximum_z_m" not in item else float(item["maximum_z_m"])
                ),
                point_robot_base_m=(
                    None
                    if "point_m" not in item
                    else tuple(float(value) for value in item["point_m"])
                ),
            )
        )

    scene_path = config_path.parent.parent / str(required(raw, "scene_path"))
    if not scene_path.is_file():
        raise ValueError(f"configured MuJoCo scene does not exist: {scene_path}")
    return MujocoFixedRgbdConfigV1(
        scene_path=scene_path,
        keyframe_name=str(required(raw, "keyframe_name")),
        camera_name=str(required(camera, "name")),
        robot_base_body_name=str(required(raw, "robot_base_body")),
        width=width,
        height=height,
        sensor_id=str(required(camera, "sensor_id")),
        calibration_id=str(required(camera, "calibration_id")),
        entity_specs=entity_specs,
        calibrated_regions=tuple(calibrated_regions),
    )
