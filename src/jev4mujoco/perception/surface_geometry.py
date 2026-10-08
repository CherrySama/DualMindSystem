from __future__ import annotations

from math import hypot

from jev4mujoco.contracts.visual_scene import Polygon2, VisualPredicate


Vec2 = tuple[float, float]


def _point_segment_distance_v1(point: Vec2, start: Vec2, end: Vec2) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1.0e-18:
        return hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    return hypot(
        point[0] - (start[0] + projection * dx),
        point[1] - (start[1] + projection * dy),
    )


def _point_on_segment_v1(point: Vec2, start: Vec2, end: Vec2) -> bool:
    cross = (point[0] - start[0]) * (end[1] - start[1]) - (
        point[1] - start[1]
    ) * (end[0] - start[0])
    if abs(cross) > 1.0e-12:
        return False
    return (
        min(start[0], end[0]) - 1.0e-12
        <= point[0]
        <= max(start[0], end[0]) + 1.0e-12
        and min(start[1], end[1]) - 1.0e-12
        <= point[1]
        <= max(start[1], end[1]) + 1.0e-12
    )


def _point_in_polygon_inclusive_v1(point: Vec2, polygon: Polygon2) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment_v1(point, previous, current):
            return True
        if (previous[1] > point[1]) != (current[1] > point[1]):
            intersection_x = (
                (current[0] - previous[0])
                * (point[1] - previous[1])
                / (current[1] - previous[1])
                + previous[0]
            )
            if point[0] <= intersection_x:
                inside = not inside
        previous = current
    return inside


def _orientation_v1(a: Vec2, b: Vec2, c: Vec2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (
        c[0] - a[0]
    )


def _segments_intersect_v1(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool:
    ab_c = _orientation_v1(a, b, c)
    ab_d = _orientation_v1(a, b, d)
    cd_a = _orientation_v1(c, d, a)
    cd_b = _orientation_v1(c, d, b)
    if (
        (ab_c > 1.0e-12 and ab_d < -1.0e-12)
        or (ab_c < -1.0e-12 and ab_d > 1.0e-12)
    ) and (
        (cd_a > 1.0e-12 and cd_b < -1.0e-12)
        or (cd_a < -1.0e-12 and cd_b > 1.0e-12)
    ):
        return True
    return any(
        abs(value) <= 1.0e-12 and _point_on_segment_v1(point, start, end)
        for value, point, start, end in (
            (ab_c, c, a, b),
            (ab_d, d, a, b),
            (cd_a, a, c, d),
            (cd_b, b, c, d),
        )
    )


def polygons_overlap_v1(first: Polygon2, second: Polygon2) -> bool:
    if any(_point_in_polygon_inclusive_v1(point, second) for point in first):
        return True
    if any(_point_in_polygon_inclusive_v1(point, first) for point in second):
        return True
    return any(
        _segments_intersect_v1(first[index - 1], point, second[j - 1], other)
        for index, point in enumerate(first)
        for j, other in enumerate(second)
    )


def polygon_contains_polygon_v1(container: Polygon2, contained: Polygon2) -> bool:
    return all(_point_in_polygon_inclusive_v1(point, container) for point in contained)


def polygon_clearance_v1(first: Polygon2, second: Polygon2) -> float:
    """Minimum planar boundary clearance; zero means overlap or touching."""
    if polygons_overlap_v1(first, second):
        return 0.0
    distances = [
        _point_segment_distance_v1(point, second[index - 1], edge_end)
        for point in first
        for index, edge_end in enumerate(second)
    ]
    distances.extend(
        _point_segment_distance_v1(point, first[index - 1], edge_end)
        for point in second
        for index, edge_end in enumerate(first)
    )
    return min(distances)


def signed_point_polygon_clearance_v1(point: Vec2, polygon: Polygon2) -> float:
    """Boundary distance: positive outside, negative inside, zero on an edge."""
    distance = min(
        _point_segment_distance_v1(point, polygon[index - 1], end)
        for index, end in enumerate(polygon)
    )
    return -distance if _point_in_polygon_inclusive_v1(point, polygon) else distance


def footprint_clear_of_region_v1(
    footprint: Polygon2,
    region_boundary: Polygon2,
    required_clearance_m: float,
) -> bool:
    if required_clearance_m < 0.0:
        raise ValueError("required_clearance_m must be nonnegative")
    if polygons_overlap_v1(footprint, region_boundary):
        return False
    return polygon_clearance_v1(footprint, region_boundary) >= required_clearance_m


def surface_push_desired_inside_v1(
    predicate: VisualPredicate,
    desired_value: bool,
) -> bool:
    if predicate is VisualPredicate.IN_REGION and desired_value:
        return True
    if predicate is VisualPredicate.CLEAR_OF_REGION and desired_value:
        return False
    raise ValueError("unsupported surface-push goal relation")


def polygon_center_v1(boundary: Polygon2) -> Vec2:
    return (
        sum(point[0] for point in boundary) / len(boundary),
        sum(point[1] for point in boundary) / len(boundary),
    )


def surface_push_direction_v1(
    subject_xy: Vec2,
    reference_boundary: Polygon2,
    desired_inside: bool,
) -> Vec2:
    """Return a goal direction, never a metric action or fixed trajectory."""
    center = polygon_center_v1(reference_boundary)
    dx = center[0] - subject_xy[0]
    dy = center[1] - subject_xy[1]
    if not desired_inside:
        dx, dy = -dx, -dy
    norm = hypot(dx, dy)
    if norm <= 1.0e-9:
        edge_midpoints = tuple(
            (
                0.5 * (reference_boundary[index - 1][0] + point[0]),
                0.5 * (reference_boundary[index - 1][1] + point[1]),
            )
            for index, point in enumerate(reference_boundary)
        )
        nearest = min(
            edge_midpoints,
            key=lambda point: hypot(
                point[0] - subject_xy[0], point[1] - subject_xy[1]
            ),
        )
        dx = nearest[0] - subject_xy[0]
        dy = nearest[1] - subject_xy[1]
        norm = hypot(dx, dy)
    if norm <= 1.0e-9:
        raise ValueError("surface push goal direction is geometrically undefined")
    return dx / norm, dy / norm


def surface_push_precontact_xy_v1(
    subject_xy: Vec2,
    subject_extent_xy: Vec2,
    direction_xy: Vec2,
    clearance_m: float,
) -> Vec2:
    projected_half_extent = 0.5 * (
        abs(direction_xy[0]) * subject_extent_xy[0]
        + abs(direction_xy[1]) * subject_extent_xy[1]
    )
    offset = projected_half_extent + clearance_m
    return (
        subject_xy[0] - direction_xy[0] * offset,
        subject_xy[1] - direction_xy[1] * offset,
    )
