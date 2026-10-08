"""已标定工具凸包及模型选择的物体侧面；不包含物体隐藏状态。"""
from dataclasses import dataclass
from math import hypot, isfinite

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ToolPartGeometryV1:
    part_id: str
    vertices_robot_base_m: tuple[Vec3, ...]
    faces: tuple[tuple[int, int, int], ...]

    def __post_init__(self):
        if not self.part_id or len(self.vertices_robot_base_m) < 4 or not self.faces:
            raise ValueError("tool part requires an identity and a convex hull")
        if any(len(v) != 3 or not all(isfinite(x) for x in v) for v in self.vertices_robot_base_m):
            raise ValueError("tool vertices must be finite XYZ points")
        if any(len(f) != 3 or any(type(i) is not int or not 0 <= i < len(self.vertices_robot_base_m)
            for i in f) for f in self.faces):
            raise ValueError("tool faces must reference three hull vertices")


@dataclass(frozen=True, slots=True)
class ToolGeometryV1:
    parts: tuple[ToolPartGeometryV1, ...]
    source: str = "calibrated_robot_model_and_measured_finger_joints"

    def __post_init__(self):
        if not self.parts or not self.source or len({p.part_id for p in self.parts}) != len(self.parts):
            raise ValueError("tool geometry requires unique parts and provenance")


@dataclass(frozen=True, slots=True)
class ContactSideV1:
    subject_ref: str
    edge_index: int
    start_xy_m: Vec2
    end_xy_m: Vec2
    outward_normal_xy: Vec2

    def __post_init__(self):
        vectors = (self.start_xy_m, self.end_xy_m, self.outward_normal_xy)
        if (not self.subject_ref or type(self.edge_index) is not int or self.edge_index < 0
            or any(len(v) != 2 or not all(isfinite(x) for x in v) for v in vectors)):
            raise ValueError("contact side requires a bound subject and finite geometry")
        dx, dy = self.end_xy_m[0] - self.start_xy_m[0], self.end_xy_m[1] - self.start_xy_m[1]
        if hypot(dx, dy) <= 1e-12 or abs(hypot(*self.outward_normal_xy) - 1) > 1e-6:
            raise ValueError("contact side must have length and a unit normal")
        if abs(dx * self.outward_normal_xy[0] + dy * self.outward_normal_xy[1]) > 1e-8:
            raise ValueError("contact normal must be perpendicular to the edge")
