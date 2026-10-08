"""仿真最终隐藏真值验收；原方法通过委托调用此处的计算。

函数接收原 Runtime 实例，保留配置、对象切换顺序及相关副作用。
这些计算不向策略或具名公开状态 verifier 提供真值。"""

from __future__ import annotations

from math import hypot, sqrt
import numpy as np
from numpy.typing import NDArray
from jev4mujoco.contracts.actions import RawStateSnapshot
from jev4mujoco.perception.surface_geometry import (
    footprint_clear_of_region_v1,
    polygon_clearance_v1,
    polygon_contains_polygon_v1,
)
from jev4mujoco.contracts.visual_scene import VisualPredicate

from jev4mujoco.experiments.results import (
    HiddenTruthResultV1,
    SurfacePushHiddenTruthResultV1,
    PickHiddenTruthResultV1,
    ObjectRegionTruthV1,
    MultiObjectHiddenTruthResultV1,
    RelativePlacementHiddenTruthResultV1,
    ArrangementHiddenTruthResultV1,
    StackHiddenTruthResultV1,
    ContainerPlacementHiddenTruthResultV1,
    ButtonPressHiddenTruthResultV1,
    OrderedSequenceHiddenTruthResultV1,
)

def _region(self, region_id: str):
    return next(
        region for region in self._visual_config.calibrated_regions
        if region.region_id == region_id
    )


def _box_bounds(self, subject_ref: str, current: RawStateSnapshot | None = None):
    corners = self.backend.object_box_corners_robot_base_m(subject_ref, current)
    return np.min(corners, axis=0), np.max(corners, axis=0)


def _box_footprint(self, subject_ref: str, current: RawStateSnapshot | None = None):
    # Convex hull of all eight projected collision corners, including tilted boxes.
    corners = self.backend.object_box_corners_robot_base_m(subject_ref, current)
    points = sorted(set(tuple(point[:2]) for point in corners))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def half_hull(sequence):
        hull = []
        for point in sequence:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
        return hull

    return tuple(half_hull(points)[:-1] + half_hull(reversed(points))[:-1])


def _table_supported(self, current, subject_ref):
    minimum, _ = _box_bounds(self, subject_ref, current)
    height = _region(self, "table_surface").surface_height_robot_base_m
    return current.contacts.bowl_table_contact and abs(minimum[2] - height) <= 0.006


def evaluate_pick_and_place(self, raw: RawStateSnapshot) -> HiddenTruthResultV1:
    footprint_inside = polygon_contains_polygon_v1(
        _region(self, "target_region").boundary_xy_robot_base_m,
        _box_footprint(self, "block_1", raw),
    )
    speed = sqrt(sum(value * value for value in raw.bowl_linear_velocity_world_mps))
    supported = _table_supported(self, raw, "block_1")
    stable = speed <= 0.01
    fingers_clear = not (
        raw.contacts.left_finger_bowl_contact
        or raw.contacts.right_finger_bowl_contact
    )
    gripper_open = (
        abs(raw.gripper_joint_m - self.config.gripper_open_joint_m)
        <= self._verifier_config.gripper_tolerance_m
    )
    checks = {
        "footprint_inside_target": footprint_inside,
        "supported": supported,
        "stable": stable,
        "fingers_clear": fingers_clear,
        "gripper_open": gripper_open,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return HiddenTruthResultV1(
        success=not failed,
        footprint_inside_target=footprint_inside,
        supported=supported,
        stable=stable,
        fingers_clear=fingers_clear,
        gripper_open=gripper_open,
        object_center_robot_base_m=raw.bowl_position_robot_base_m,
        object_linear_speed_mps=speed,
        reason="all hidden checks passed" if not failed else ", ".join(failed),
    )


def evaluate_multi_object_regions(
    self, destinations: dict[str, str]
) -> MultiObjectHiddenTruthResultV1:
    regions = {
        region.region_id: region
        for region in self._visual_config.calibrated_regions
    }
    results: list[ObjectRegionTruthV1] = []
    for subject_ref, target_ref in destinations.items():
        current = self.backend.select_active_object(subject_ref)
        inside = polygon_contains_polygon_v1(
            regions[target_ref].boundary_xy_robot_base_m,
            _box_footprint(self, subject_ref),
        )
        speed = sqrt(
            sum(value * value for value in current.bowl_linear_velocity_world_mps)
        )
        results.append(
            ObjectRegionTruthV1(
                subject_ref=subject_ref,
                target_ref=target_ref,
                footprint_inside_target=inside,
                supported=_table_supported(self, current, subject_ref),
                stable=speed <= 0.01,
                fingers_clear=not (
                    current.contacts.left_finger_bowl_contact
                    or current.contacts.right_finger_bowl_contact
                ),
                object_center_robot_base_m=current.bowl_position_robot_base_m,
            )
        )
    final = self.backend.snapshot()
    gripper_open = (
        abs(final.gripper_joint_m - self.config.gripper_open_joint_m)
        <= self._verifier_config.gripper_tolerance_m
    )
    failed = [
        item.subject_ref
        for item in results
        if not (
            item.footprint_inside_target
            and item.supported
            and item.stable
            and item.fingers_clear
        )
    ]
    if not gripper_open:
        failed.append("gripper_open")
    return MultiObjectHiddenTruthResultV1(
        success=not failed,
        objects=tuple(results),
        gripper_open=gripper_open,
        reason=(
            "all hidden checks passed"
            if not failed
            else f"failed hidden checks: {', '.join(failed)}"
        ),
    )


def quaternion_rotation(quaternion_wxyz: tuple[float, float, float, float]) -> NDArray[np.float64]:
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def evaluate_pick(self, raw: RawStateSnapshot) -> PickHiddenTruthResultV1:
    both_contacts = (
        raw.contacts.left_finger_bowl_contact
        and raw.contacts.right_finger_bowl_contact
    )
    gripper_blocked = raw.gripper_joint_m >= 0.003
    gripper_commanded_closed = (
        abs(raw.gripper_target_m - self.config.gripper_closed_joint_m)
        <= self._verifier_config.gripper_tolerance_m
    )
    object_near_tcp = sqrt(
        sum(
            (object_value - tcp_value) ** 2
            for object_value, tcp_value in zip(
                raw.bowl_position_robot_base_m,
                raw.tcp_position_robot_base_m,
            )
        )
    ) <= 0.05
    speed = sqrt(sum(value * value for value in raw.bowl_linear_velocity_world_mps))
    stable = speed <= 0.01
    checks = {
        "both_fingers_contact": both_contacts,
        "gripper_blocked": gripper_blocked,
        "gripper_commanded_closed": gripper_commanded_closed,
        "object_near_tcp": object_near_tcp,
        "stable": stable,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return PickHiddenTruthResultV1(
        success=not failed,
        both_fingers_contact=both_contacts,
        gripper_blocked=gripper_blocked,
        gripper_commanded_closed=gripper_commanded_closed,
        object_near_tcp=object_near_tcp,
        stable=stable,
        object_center_robot_base_m=raw.bowl_position_robot_base_m,
        object_linear_speed_mps=speed,
        reason="all hidden checks passed" if not failed else ", ".join(failed),
    )


def evaluate_surface_push(
    self, raw: RawStateSnapshot
) -> SurfacePushHiddenTruthResultV1:
    footprint = _box_footprint(self, "block_1", raw)
    target_boundary = _region(self, "target_region").boundary_xy_robot_base_m
    footprint_inside = polygon_contains_polygon_v1(target_boundary, footprint)
    required_clearance = (
        self._verifier_config.surface_push_clear_of_region_margin_m
    )
    target_clearance = polygon_clearance_v1(footprint, target_boundary)
    footprint_clear = footprint_clear_of_region_v1(
        footprint,
        target_boundary,
        required_clearance,
    )
    speed = sqrt(sum(value * value for value in raw.bowl_linear_velocity_world_mps))
    supported = _table_supported(self, raw, "block_1")
    stable = speed <= 0.01
    effector_clear = not (
        raw.contacts.left_finger_bowl_contact
        or raw.contacts.right_finger_bowl_contact
    )
    goal_relation_satisfied = (
        footprint_inside if self._desired_inside_target else footprint_clear
    )
    checks = {
        "goal_relation_satisfied": goal_relation_satisfied,
        "supported": supported,
        "stable": stable,
        "effector_clear": effector_clear,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return SurfacePushHiddenTruthResultV1(
        success=not failed,
        footprint_inside_target=footprint_inside,
        footprint_clear_of_target=footprint_clear,
        target_clearance_m=target_clearance,
        required_clearance_m=required_clearance,
        supported=supported,
        stable=stable,
        effector_clear=effector_clear,
        object_center_robot_base_m=raw.bowl_position_robot_base_m,
        object_linear_speed_mps=speed,
        reason="all hidden checks passed" if not failed else ", ".join(failed),
    )


def evaluate_sort(self, raw: RawStateSnapshot) -> MultiObjectHiddenTruthResultV1:
    del raw
    return self._multi_object_region_hidden_truth(self._sort_destinations)


def evaluate_clear(self, raw: RawStateSnapshot) -> MultiObjectHiddenTruthResultV1:
    del raw
    return self._multi_object_region_hidden_truth(self._clear_destinations)


def evaluate_relative_placement(
    self, raw: RawStateSnapshot
) -> RelativePlacementHiddenTruthResultV1:
    del raw
    subject = self.backend.select_active_object("block_1")
    reference = self.backend.select_active_object("block_2")
    subject_min, _ = _box_bounds(self, "block_1")
    _, reference_max = _box_bounds(self, "block_2")
    relation_satisfied = (
        subject_min[1] >= reference_max[1]
        + self._verifier_config.visual_relation_config.directional_gap_m
    )
    subject_speed = sqrt(
        sum(value * value for value in subject.bowl_linear_velocity_world_mps)
    )
    reference_speed = sqrt(
        sum(value * value for value in reference.bowl_linear_velocity_world_mps)
    )
    supported = _table_supported(self, subject, "block_1")
    stable = subject_speed <= 0.01
    reference_supported = _table_supported(self, reference, "block_2")
    reference_stable = reference_speed <= 0.01
    reference_displacement = sqrt(
        sum(
            (
                reference.bowl_position_robot_base_m[index]
                - self._initial_reference_center[index]
            )
            ** 2
            for index in range(3)
        )
    )
    reference_displacement_within_limit = (
        reference_displacement
        <= self._verifier_config.placement_reference_maximum_displacement_m
    )
    fingers_clear = not (
        subject.contacts.left_finger_bowl_contact
        or subject.contacts.right_finger_bowl_contact
    )
    gripper_open = (
        abs(reference.gripper_joint_m - self.config.gripper_open_joint_m)
        <= self._verifier_config.gripper_tolerance_m
    )
    checks = {
        "left_of": relation_satisfied,
        "supported": supported,
        "stable": stable,
        "reference_supported": reference_supported,
        "reference_stable": reference_stable,
        "reference_displacement_within_limit": (
            reference_displacement_within_limit
        ),
        "fingers_clear": fingers_clear,
        "gripper_open": gripper_open,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return RelativePlacementHiddenTruthResultV1(
        success=not failed,
        relation=VisualPredicate.LEFT_OF.value,
        relation_satisfied=relation_satisfied,
        supported=supported,
        stable=stable,
        reference_supported=reference_supported,
        reference_stable=reference_stable,
        reference_displacement_m=reference_displacement,
        reference_displacement_within_limit=(
            reference_displacement_within_limit
        ),
        fingers_clear=fingers_clear,
        gripper_open=gripper_open,
        subject_center_robot_base_m=subject.bowl_position_robot_base_m,
        reference_center_robot_base_m=reference.bowl_position_robot_base_m,
        reason="all hidden checks passed" if not failed else ", ".join(failed),
    )


def evaluate_arrangement(self, raw: RawStateSnapshot) -> ArrangementHiddenTruthResultV1:
    del raw
    base = self._multi_object_region_hidden_truth(
        self._arrangement_destinations
    )
    centers: list[tuple[float, float, float]] = []
    footprints = []
    for subject_ref in self._arrangement_destinations:
        current = self.backend.select_active_object(subject_ref)
        centers.append(current.bowl_position_robot_base_m)
        footprints.append(_box_footprint(self, subject_ref))
    row_cross_span = max(center[1] for center in centers) - min(
        center[1] for center in centers
    )
    minimum_clearance = min(
        polygon_clearance_v1(footprints[index], footprints[other])
        for index in range(len(footprints))
        for other in range(index + 1, len(footprints))
    )
    row_passed = (
        row_cross_span
        <= self._verifier_config.arrangement_row_cross_axis_tolerance_m
    )
    clearance_passed = (
        minimum_clearance
        >= self._verifier_config.arrangement_minimum_pairwise_clearance_m
    )
    failed = []
    if not base.success:
        failed.append(base.reason)
    if not row_passed:
        failed.append("row_cross_axis_span")
    if not clearance_passed:
        failed.append("pairwise_clearance")
    return ArrangementHiddenTruthResultV1(
        success=not failed,
        objects=base.objects,
        row_cross_axis_span_m=row_cross_span,
        minimum_pairwise_clearance_m=minimum_clearance,
        gripper_open=base.gripper_open,
        reason="all hidden checks passed" if not failed else ", ".join(failed),
    )


def evaluate_stack(self, raw: RawStateSnapshot) -> StackHiddenTruthResultV1:
    del raw
    top = self.backend.select_active_object(self._top_block_ref)
    base = self.backend.select_active_object(self._base_block_ref)
    top_min, top_max = _box_bounds(self, self._top_block_ref)
    base_min, base_max = _box_bounds(self, self._base_block_ref)
    top_min_x, top_min_y, top_bottom = top_min
    top_max_x, top_max_y, _ = top_max
    base_min_x, base_min_y, _ = base_min
    base_max_x, base_max_y, base_top = base_max
    overlap_x = max(0.0, min(top_max_x, base_max_x) - max(top_min_x, base_min_x))
    overlap_y = max(0.0, min(top_max_y, base_max_y) - max(top_min_y, base_min_y))
    top_area = (top_max_x - top_min_x) * (top_max_y - top_min_y)
    overlap_fraction = 0.0 if top_area <= 0.0 else overlap_x * overlap_y / top_area
    on_surface = (
        abs(top_bottom - base_top)
        <= self._verifier_config.visual_relation_config.support_height_tolerance_m
        and overlap_fraction
        >= self._verifier_config.visual_relation_config.support_minimum_overlap_fraction
    )
    top_speed = sqrt(
        sum(value * value for value in top.bowl_linear_velocity_world_mps)
    )
    base_speed = sqrt(
        sum(value * value for value in base.bowl_linear_velocity_world_mps)
    )
    base_displacement = sqrt(
        sum(
            (
                base.bowl_position_robot_base_m[index]
                - self._initial_base_center[index]
            )
            ** 2
            for index in range(3)
        )
    )
    checks = {
        "on_surface": on_surface,
        "top_supported": top.contacts.bowl_table_contact,
        "top_stable": top_speed <= 0.01,
        "base_supported": _table_supported(self, base, self._base_block_ref),
        "base_stable": base_speed <= 0.01,
        "base_displacement_within_limit": (
            base_displacement
            <= self._verifier_config.placement_reference_maximum_displacement_m
        ),
        "fingers_clear": not (
            top.contacts.left_finger_bowl_contact
            or top.contacts.right_finger_bowl_contact
        ),
        "gripper_open": (
            abs(base.gripper_joint_m - self.config.gripper_open_joint_m)
            <= self._verifier_config.gripper_tolerance_m
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return StackHiddenTruthResultV1(
        success=not failed,
        on_surface=on_surface,
        support_overlap_fraction=overlap_fraction,
        top_supported=checks["top_supported"],
        top_stable=checks["top_stable"],
        base_supported=checks["base_supported"],
        base_stable=checks["base_stable"],
        base_displacement_m=base_displacement,
        base_displacement_within_limit=checks["base_displacement_within_limit"],
        fingers_clear=checks["fingers_clear"],
        gripper_open=checks["gripper_open"],
        top_center_robot_base_m=top.bowl_position_robot_base_m,
        base_center_robot_base_m=base.bowl_position_robot_base_m,
        reason="all hidden checks passed" if not failed else ", ".join(failed),
    )


def evaluate_inside_bowl(
    self, raw: RawStateSnapshot
) -> ContainerPlacementHiddenTruthResultV1:
    interior = _region(self, "bowl_1_interior")
    minimum, maximum = _box_bounds(self, "block_1", raw)
    fully_inside = (
        polygon_contains_polygon_v1(
            interior.boundary_xy_robot_base_m, _box_footprint(self, "block_1", raw)
        )
        and minimum[2] >= interior.minimum_z_robot_base_m - 0.006
        and maximum[2] <= interior.maximum_z_robot_base_m + 0.006
    )
    speed = sqrt(sum(value * value for value in raw.bowl_linear_velocity_world_mps))
    supported = (
        raw.contacts.bowl_table_contact
        and abs(minimum[2] - interior.surface_height_robot_base_m) <= 0.008
    )
    stable = speed <= 0.01
    fingers_clear = not (
        raw.contacts.left_finger_bowl_contact
        or raw.contacts.right_finger_bowl_contact
    )
    gripper_open = (
        abs(raw.gripper_joint_m - self.config.gripper_open_joint_m)
        <= self._verifier_config.gripper_tolerance_m
    )
    checks = {
        "fully_inside": fully_inside,
        "supported_by_bowl": supported,
        "stable": stable,
        "fingers_clear": fingers_clear,
        "gripper_open": gripper_open,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return ContainerPlacementHiddenTruthResultV1(
        success=not failed,
        fully_inside=fully_inside,
        supported_by_container_floor=supported,
        stable=stable,
        fingers_clear=fingers_clear,
        gripper_open=gripper_open,
        object_center_robot_base_m=raw.bowl_position_robot_base_m,
        object_linear_speed_mps=speed,
        reason="all hidden checks passed" if not failed else ", ".join(failed),
    )


def evaluate_button_press(self, raw: RawStateSnapshot) -> ButtonPressHiddenTruthResultV1:
    travel = self._initial_button_center_z - raw.bowl_position_robot_base_m[2]
    travel_reached = travel >= self._verifier_config.press_minimum_travel_m
    effector_contact = (
        raw.contacts.left_finger_bowl_contact
        or raw.contacts.right_finger_bowl_contact
    )
    supported = hypot(
        raw.bowl_position_robot_base_m[0] - self._initial_button_center_xy[0],
        raw.bowl_position_robot_base_m[1] - self._initial_button_center_xy[1],
    ) <= 0.001
    speed = sqrt(sum(value * value for value in raw.bowl_linear_velocity_world_mps))
    stable = speed <= 0.01
    gripper_closed = (
        abs(
            raw.gripper_joint_m
            - self._verifier_config.surface_push_closed_gripper_position_m
        )
        <= self._verifier_config.surface_push_gripper_position_tolerance_m
        and abs(
            raw.gripper_target_m
            - self._verifier_config.surface_push_closed_gripper_position_m
        )
        <= self._verifier_config.surface_push_gripper_position_tolerance_m
    )
    checks = {
        "travel_reached": travel_reached,
        "effector_contact": effector_contact,
        "button_supported": supported,
        "button_stable": stable,
        "gripper_closed": gripper_closed,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return ButtonPressHiddenTruthResultV1(
        success=not failed,
        press_travel_m=travel,
        travel_reached=travel_reached,
        effector_contact=effector_contact,
        button_supported=supported,
        button_stable=stable,
        gripper_closed=gripper_closed,
        reason="all hidden checks passed" if not failed else ", ".join(failed),
    )


def evaluate_ordered_sequence(
    self, raw: RawStateSnapshot
) -> OrderedSequenceHiddenTruthResultV1:
    del raw
    base = self._multi_object_region_hidden_truth(self._ordered_destinations)
    failed = [
        item.subject_ref
        for item in base.objects
        if not (
            item.footprint_inside_target
            and item.supported
            and item.stable
            and item.fingers_clear
        )
    ]
    return OrderedSequenceHiddenTruthResultV1(
        success=not failed,
        objects=base.objects,
        stage_order=("place_stage", "push_stage"),
        reason=(
            "all hidden checks passed"
            if not failed
            else f"failed hidden checks: {', '.join(failed)}"
        ),
    )
