"""从机器人标定与公开反馈计算工具—有限侧面几何；不推断物理接触。"""
from dataclasses import dataclass, asdict
from itertools import combinations
from math import hypot

import mujoco
import numpy as np

from jev4mujoco.contracts.tool_geometry import ContactSideV1, ToolGeometryV1, ToolPartGeometryV1
from jev4mujoco.contracts.visual_scene import Visibility


@dataclass(frozen=True, slots=True)
class _CalibratedPart:
    name: str
    vertices_tcp_m: tuple
    faces: tuple
    slide_axis_tcp: tuple
    right: bool


class CalibratedFingerToolV1:
    def __init__(self, parts):
        self.parts = tuple(parts)

    @classmethod
    def from_model(cls, model, tcp_name="tcp"):
        """Read only robot CAD/kinematics from the static model, never live object data.

        Convex graph layout: MuJoCo APItypes.html#convex-hulls.
        """
        def named(kind, name):
            index = mujoco.mj_name2id(model, kind, name)
            if index < 0:
                raise ValueError(f"missing calibrated robot geometry: {name}")
            return index

        tcp_id = named(mujoco.mjtObj.mjOBJ_SITE, tcp_name)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        tcp_rotation = data.site_xmat[tcp_id].reshape(3, 3)
        parts = []
        for name, joint_name, right in (("gripper_L_geom", "gripper_L_joint", False),
            ("gripper_R_geom", "gripper_R_joint", True)):
            geom = named(mujoco.mjtObj.mjOBJ_GEOM, name)
            joint = named(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE:
                raise ValueError("finger calibration supports slide joints only")
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            mesh = int(model.geom_dataid[geom])
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or mesh < 0:
                raise ValueError("finger calibration requires a convex mesh")
            address = int(model.mesh_graphadr[mesh])
            if address < 0:
                raise ValueError("finger mesh has no calibrated convex hull")
            graph = model.mesh_graph[address:]
            nv, nf = int(graph[0]), int(graph[1])
            vertex_ids = graph[2 + nv:2 + 2 * nv]
            face_ids = graph[2 + 3 * nv + 3 * nf:2 + 3 * nv + 6 * nf].reshape(-1, 3)
            mapping = {int(global_id): local_id for local_id, global_id in enumerate(vertex_ids)}
            faces = tuple(tuple(mapping[int(i)] for i in face) for face in face_ids)
            vertices = np.asarray(model.mesh_vert[model.mesh_vertadr[mesh] + vertex_ids], dtype=float)
            world = np.einsum("ij,kj->ki", data.geom_xmat[geom].reshape(3, 3), vertices) + data.geom_xpos[geom]
            local = np.einsum("ij,ki->kj", tcp_rotation, world - data.site_xpos[tcp_id])
            body_rotation = data.xmat[model.jnt_bodyid[joint]].reshape(3, 3)
            axis_world = np.einsum("ij,j->i", body_rotation, model.jnt_axis[joint])
            axis = np.einsum("ij,i->j", tcp_rotation, axis_world)
            # Remove the static model's reference joint position from the calibration.
            local -= float(data.qpos[model.jnt_qposadr[joint]]) * axis
            parts.append(_CalibratedPart(name, tuple(map(tuple, local)), faces, tuple(axis), right))
        if not parts:
            raise ValueError("no enabled finger geometry")
        return cls(parts)

    def observe(self, tcp, quaternion, left_joint, right_joint):
        if right_joint is None:
            return None  # Independent right-finger feedback is required, never assumed symmetric.
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
        parts = []
        for part in self.parts:
            joint = right_joint if part.right else left_joint
            local = np.asarray(part.vertices_tcp_m) + joint * np.asarray(part.slide_axis_tcp)
            points = np.einsum("ij,kj->ki", rotation.reshape(3, 3), local) + np.asarray(tcp)
            parts.append(ToolPartGeometryV1(part.name, tuple(map(tuple, points)), part.faces))
        return ToolGeometryV1(tuple(parts))


def observed_contact_sides_v1(snapshot):
    entity = next((e for e in snapshot.visual_scene.entities if e.track_id == snapshot.skill.subject_ref), None)
    if entity is None or entity.visibility is not Visibility.CLEAR:
        return ()
    footprint = entity.geometry.footprint_xy_robot_base_m.value
    if footprint is None:
        return ()
    area2 = sum(footprint[i - 1][0] * p[1] - p[0] * footprint[i - 1][1]
        for i, p in enumerate(footprint))
    if abs(area2) <= 1e-12:
        return ()
    orientation = 1 if area2 > 0 else -1
    sides = []
    for index, end in enumerate(footprint):
        start = footprint[index - 1]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = hypot(dx, dy)
        if length > 1e-12:
            sides.append(ContactSideV1(entity.track_id, index, start, end,
                (orientation * dy / length, -orientation * dx / length)))
    return tuple(sides)


def _clip_polygon(points, axis, bound, above):
    result = []
    if not points:
        return result
    previous = points[-1]
    for point in points:
        inside = point[axis] >= bound if above else point[axis] <= bound
        was_inside = previous[axis] >= bound if above else previous[axis] <= bound
        if inside != was_inside:
            fraction = (bound - previous[axis]) / (point[axis] - previous[axis])
            result.append(previous + fraction * (point - previous))
        if inside:
            result.append(point)
        previous = point
    return result


def _convex_closest_points(vertices, patch):
    """GJK distance with witnesses; the patch is a finite, possibly flat hull.

    The simplex projection enumerates its admissible faces (at most four points
    in 3D). Numerical tolerance is not a contact or task acceptance threshold.
    Return unknown if the bounded calculation cannot certify convergence.
    """
    tolerance = 1e-10

    def support(direction):
        return (vertices[np.argmax(vertices @ direction)],
            patch[np.argmin(patch @ direction)])

    def project(simplex):
        differences = np.asarray([p - q for p, q in simplex])
        best = None
        for count in range(1, min(4, len(simplex)) + 1):
            for indices in combinations(range(len(simplex)), count):
                points = differences[list(indices)]
                if count == 1:
                    weights = np.ones(1)
                else:
                    coefficients = np.linalg.lstsq((points[:-1] - points[-1]).T,
                        -points[-1], rcond=1e-12)[0]
                    weights = np.append(coefficients, 1 - coefficients.sum())
                if weights.min() < -1e-12:
                    continue
                weights = np.maximum(weights, 0)
                weights /= weights.sum()
                closest = weights @ points
                norm2 = float(closest @ closest)
                if best is None or norm2 < best[0]:
                    best = (norm2, indices, weights, closest)
        _, indices, weights, closest = best
        reduced = [simplex[i] for i, weight in zip(indices, weights) if weight > 0]
        tool_point = sum(weight * simplex[i][0] for i, weight in zip(indices, weights))
        side_point = sum(weight * simplex[i][1] for i, weight in zip(indices, weights))
        return reduced, closest, tool_point, side_point

    direction = vertices.mean(axis=0) - patch.mean(axis=0)
    if np.linalg.norm(direction) <= tolerance:
        direction = np.array((1.0, 0.0, 0.0))
    simplex = [support(direction)]
    for _ in range(64):
        simplex, closest, tool_point, side_point = project(simplex)
        distance = float(np.linalg.norm(closest))
        if distance <= tolerance:
            return 0.0, tool_point, side_point
        next_pair = support(-closest)
        # Support bounds the remaining improvement in the closest distance.
        dual_gap = float(closest @ closest - closest @ (next_pair[0] - next_pair[1]))
        if dual_gap <= tolerance * max(distance, 1e-3):
            return distance, tool_point, side_point
        if any(np.array_equal(next_pair[0], p) and np.array_equal(next_pair[1], q) for p, q in simplex):
            return None
        simplex.append(next_pair)
    return None


def contact_geometry_v1(snapshot, side, delta=(0.0, 0.0, 0.0)):
    """Closest distance to a finite side, plus signed gap/overlap diagnostics.

    Geometry is a convex-model estimate. Zero/negative gap never confirms contact.
    """
    tool = snapshot.robot.tool_geometry
    entity = next((e for e in snapshot.visual_scene.entities if e.track_id == side.subject_ref), None)
    live_side = next((s for s in observed_contact_sides_v1(snapshot) if s.edge_index == side.edge_index), None)
    if (side.subject_ref != snapshot.skill.subject_ref or tool is None or entity is None or live_side is None
        or entity.visibility is not Visibility.CLEAR or not snapshot.freshness.visual_is_fresh):
        return {"observable": False, "reason": "fresh complete subject and calibrated tool feedback required",
            "local_error_m": None, "parts": []}
    side = live_side
    center, extent = entity.geometry.centroid_robot_base_m.value, entity.geometry.extent_m.value
    if center is None or extent is None:
        return {"observable": False, "reason": "subject height is unknown", "local_error_m": None, "parts": []}
    length = hypot(side.end_xy_m[0] - side.start_xy_m[0], side.end_xy_m[1] - side.start_xy_m[1])
    tangent = np.array(((side.end_xy_m[0] - side.start_xy_m[0]) / length,
        (side.end_xy_m[1] - side.start_xy_m[1]) / length))
    normal = np.asarray(side.outward_normal_xy)
    low, high = center[2] - extent[2] / 2, center[2] + extent[2] / 2
    patch = np.asarray([(x, y, z) for x, y in (side.start_xy_m, side.end_xy_m) for z in (low, high)])
    rows = []
    for part in tool.parts:
        vertices = np.asarray(part.vertices_robot_base_m) + np.asarray(delta)
        offset = vertices[:, :2] - np.asarray(side.start_xy_m)
        coords = np.column_stack((np.einsum("ij,j->i", offset, normal),
            np.einsum("ij,j->i", offset, tangent), vertices[:, 2]))
        tangent_deficit = max(float(coords[:, 1].min()) - length, -float(coords[:, 1].max()), 0)
        height_deficit = max(float(coords[:, 2].min()) - high, low - float(coords[:, 2].max()), 0)
        clipped = []
        for face in part.faces:
            polygon = [coords[index] for index in face]
            for axis, bound, above in ((1, 0, True), (1, length, False), (2, low, True), (2, high, False)):
                polygon = _clip_polygon(polygon, axis, bound, above)
            clipped.extend(polygon)
        if clipped:
            near, far = min(float(p[0]) for p in clipped), max(float(p[0]) for p in clipped)
            gap = near if near > 0 else far if far < 0 else 0.0
            side_status = "outside" if near > 0 else "behind" if far < 0 else "intersecting"
        else:
            gap, side_status = None, "no_finite_patch_overlap"
        closest = _convex_closest_points(vertices, patch)
        error, tool_point, side_point = (None, None, None) if closest is None else closest
        rows.append({"part_id": part.part_id, "normal_gap_m": gap,
            "tangent_deficit_m": tangent_deficit, "height_deficit_m": height_deficit,
            "finite_patch_overlap": gap is not None, "local_error_m": error,
            "closest_tool_point_robot_base_m": None if tool_point is None else tuple(tool_point),
            "closest_side_point_robot_base_m": None if side_point is None else tuple(side_point),
            "correction_robot_base_m": None if error is None else tuple(side_point - tool_point)})
        rows[-1]["approach_side_status"] = side_status
    nearest = min(rows, key=lambda row: row["local_error_m"]) if all(
        row["local_error_m"] is not None for row in rows) else None
    return {"observable": True, "source": tool.source, "side": asdict(side),
        "local_error_m": None if nearest is None else nearest["local_error_m"], "parts": rows,
        "correction_robot_base_m": None if nearest is None else nearest["correction_robot_base_m"],
        "is_contact_confirmation": False, "path_verified": False}
