"""公开轮廓沿给定方向越过分离平面的充分条件；不选择推动方向。"""
from math import hypot, isfinite

from jev4mujoco.perception.surface_geometry import (
    footprint_clear_of_region_v1, polygon_clearance_v1, polygon_contains_polygon_v1,
)


def directional_exit_geometry_v1(footprint, boundary, direction_xy, margin_m, support_boundary=None):
    if (len(direction_xy) != 2 or not all(isfinite(v) for v in direction_xy)
        or abs(hypot(*direction_xy) - 1.0) > 1e-6
        or not isfinite(margin_m) or margin_m < 0):
        raise ValueError("exit geometry requires a unit direction and nonnegative margin")
    projection = lambda point: sum(point[i] * direction_xy[i] for i in (0, 1))
    plane = max(map(projection, boundary))
    rear = min(map(projection, footprint))
    remaining = max(0.0, plane + margin_m - rear)
    delta = tuple(remaining * value for value in direction_xy)
    endpoint = tuple(tuple(point[i] + delta[i] for i in (0, 1)) for point in footprint)
    return {"direction_xy": direction_xy, "region_plane_projection_m": plane,
        "subject_rear_projection_m": rear, "remaining_translation_m": remaining,
        "assumed_exit_delta_xy_m": delta,
        "endpoint_clearance_m": polygon_clearance_v1(endpoint, boundary),
        "endpoint_original_goal_satisfied": footprint_clear_of_region_v1(endpoint, boundary, margin_m),
        "endpoint_within_source_boundary": None if support_boundary is None
            else polygon_contains_polygon_v1(support_boundary, endpoint),
        "endpoint_is_counterfactual": True, "is_physical_push_prediction": False,
        "path_verification": "unsupported"}
