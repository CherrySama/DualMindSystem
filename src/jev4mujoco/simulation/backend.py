from __future__ import annotations

from collections.abc import Mapping
from threading import get_ident
from typing import TYPE_CHECKING, Callable

import mujoco
import numpy as np
from numpy.typing import NDArray

from jev4mujoco.simulation.config import StageConfig
from jev4mujoco.contracts.actions import PhysicalContactSnapshot, RawStateSnapshot

if TYPE_CHECKING:
    from jev4mujoco.perception.fixed_rgbd import (
        FixedRgbdCaptureV1,
        MujocoFixedRgbdVisualSceneBuilderV1,
    )


FloatArray = NDArray[np.float64]


class MujocoBackend:
    """Owns all access to one MuJoCo model/data pair on one thread."""

    def __init__(
        self,
        config: StageConfig,
        bowl_initial_position_world_m: tuple[float, float, float] | None = None,
        frame_sink: Callable[[NDArray[np.uint8]], None] | None = None,
        video_fps: float = 20.0,
        object_body_names: tuple[str, ...] | None = None,
        object_initial_positions_world_m: Mapping[str, tuple[float, float, float]] | None = None,
        additional_support_geom_names: tuple[str, ...] = (),
        additional_effector_geom_names: tuple[str, ...] = (),
    ):
        self.config = config
        self._owner_thread_id = get_ident()
        initial_position = (
            None
            if bowl_initial_position_world_m is None
            else np.asarray(bowl_initial_position_world_m, dtype=np.float64)
        )
        if initial_position is not None and (
            initial_position.shape != (3,) or not np.isfinite(initial_position).all()
        ):
            raise ValueError("Bowl initial position must be a finite world-frame XYZ vector")
        self._bowl_initial_position_world_m = initial_position
        self._model = mujoco.MjModel.from_xml_path(str(config.scene_path))
        self._data = mujoco.MjData(self._model)
        if not np.isclose(self._model.opt.timestep, config.physics_dt_s, atol=1e-12):
            raise ValueError(
                f"Scene timestep {self._model.opt.timestep} does not match config {config.physics_dt_s}"
            )

        self._base_body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, config.robot_base_body)
        object_names = object_body_names or (config.bowl_body,)
        if not object_names or len(object_names) != len(set(object_names)):
            raise ValueError("object_body_names must be non-empty and unique")
        if config.bowl_body not in object_names:
            raise ValueError("configured bowl_body must be one of object_body_names")
        self._object_metadata = {
            name: self._object_metadata_for(name) for name in object_names
        }
        self._object_initial_positions_world_m = {
            name: np.asarray(position, dtype=np.float64)
            for name, position in (object_initial_positions_world_m or {}).items()
        }
        unknown_initial_objects = (
            self._object_initial_positions_world_m.keys()
            - self._object_metadata.keys()
        )
        if unknown_initial_objects:
            raise ValueError(f"unknown initial object bodies: {sorted(unknown_initial_objects)}")
        if any(
            position.shape != (3,) or not np.isfinite(position).all()
            for position in self._object_initial_positions_world_m.values()
        ):
            raise ValueError("initial object positions must be finite world-frame XYZ vectors")
        self._all_object_geom_ids = frozenset(
            geom_id
            for metadata in self._object_metadata.values()
            for geom_id in metadata[3]
        )
        self._active_object_body = config.bowl_body
        self._select_object_metadata(config.bowl_body)
        self._target_body_id = (
            None if config.target_body is None
            else self._named_id(mujoco.mjtObj.mjOBJ_BODY, config.target_body)
        )
        self._left_finger_geom_id = self._named_id(
            mujoco.mjtObj.mjOBJ_GEOM, "gripper_L_geom"
        )
        self._right_finger_geom_id = self._named_id(
            mujoco.mjtObj.mjOBJ_GEOM, "gripper_R_geom"
        )
        self._table_geom_id = self._named_id(mujoco.mjtObj.mjOBJ_GEOM, "tabletop")
        self._additional_support_geom_ids = frozenset(
            self._named_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in additional_support_geom_names
        )
        self._additional_effector_geom_ids = frozenset(
            self._named_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in additional_effector_geom_names
        )
        self._tcp_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, config.tcp_site)
        self._keyframe_id = self._named_id(mujoco.mjtObj.mjOBJ_KEY, config.keyframe_name)
        self._arm_joint_ids = np.asarray(
            [self._named_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}") for index in range(1, 7)],
            dtype=np.int32,
        )
        self._arm_qpos_addresses = self._model.jnt_qposadr[self._arm_joint_ids].copy()
        self._arm_dof_addresses = self._model.jnt_dofadr[self._arm_joint_ids].copy()
        self._arm_actuator_ids = np.asarray(
            [
                self._named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{index}_position")
                for index in range(1, 7)
            ],
            dtype=np.int32,
        )
        self._arm_position_gains = self._model.actuator_gainprm[
            self._arm_actuator_ids, 0
        ].copy()
        self._arm_actuator_gears = self._model.actuator_gear[
            self._arm_actuator_ids, 0
        ].copy()
        if np.any(self._arm_position_gains <= 0.0):
            raise ValueError("Arm position actuators must have positive gains")
        if not np.allclose(self._arm_actuator_gears, 1.0, atol=1e-12):
            raise ValueError("Bias compensation currently requires unit arm actuator gears")
        gripper_joint_id = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "gripper_L_joint")
        self._gripper_qpos_address = int(self._model.jnt_qposadr[gripper_joint_id])
        right_joint_id = self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "gripper_R_joint")
        self._right_gripper_qpos_address = int(self._model.jnt_qposadr[right_joint_id])
        self._gripper_actuator_id = self._named_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_position"
        )
        gripper_gain = float(
            self._model.actuator_gainprm[self._gripper_actuator_id, 0]
        )
        if not np.isclose(gripper_gain, config.gripper_position_kp, atol=1e-12):
            raise ValueError(
                f"Scene gripper kp {gripper_gain} does not match config "
                f"{config.gripper_position_kp}"
            )
        if not self._model.actuator_forcelimited[self._gripper_actuator_id]:
            raise ValueError("Scene gripper actuator must enable force limiting")
        gripper_force_range = self._model.actuator_forcerange[
            self._gripper_actuator_id
        ]
        expected_force_range = np.asarray(
            [-config.gripper_effort_limit_n, config.gripper_effort_limit_n]
        )
        if not np.allclose(gripper_force_range, expected_force_range, atol=1e-12):
            raise ValueError(
                f"Scene gripper force range {gripper_force_range.tolist()} does not "
                f"match config {expected_force_range.tolist()}"
            )
        self._state_id = 0
        self._action_epoch = 0
        self._frame_sink = frame_sink
        self._video_period_s = 1.0 / video_fps
        self._next_frame_time_s = 0.0
        self._renderer: mujoco.Renderer | None = None
        self.reset()
        if frame_sink is not None:
            self._renderer = mujoco.Renderer(self._model, height=544, width=960)
            self._capture_frame(force=True)

    def _assert_owner(self) -> None:
        if get_ident() != self._owner_thread_id:
            raise RuntimeError("MjData may only be accessed from the backend owner thread")

    def _named_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self._model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _object_metadata_for(
        self, body_name: str
    ) -> tuple[int, int, int, frozenset[int]]:
        body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        free_joint_id = self._named_id(
            mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_free"
        )
        geom_ids = frozenset(
            geom_id
            for geom_id in range(self._model.ngeom)
            if (
                mujoco.mj_id2name(
                    self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                or ""
            ).startswith(f"{body_name}_collision_")
        )
        if not geom_ids:
            raise ValueError(f"No collision geoms found for {body_name}")
        return (
            body_id,
            int(self._model.jnt_qposadr[free_joint_id]),
            int(self._model.jnt_dofadr[free_joint_id]),
            geom_ids,
        )

    def _select_object_metadata(self, body_name: str) -> None:
        (
            self._bowl_body_id,
            self._bowl_qpos_address,
            self._bowl_dof_address,
            self._bowl_geom_ids,
        ) = self._object_metadata[body_name]

    @property
    def object_body_names(self) -> tuple[str, ...]:
        return tuple(self._object_metadata)

    def select_active_object(self, body_name: str) -> RawStateSnapshot:
        """Select which object is projected through the legacy single-object raw state."""
        self._assert_owner()
        if body_name not in self._object_metadata:
            raise ValueError(f"unknown runtime object body {body_name}")
        if body_name != self._active_object_body:
            self._active_object_body = body_name
            self._select_object_metadata(body_name)
            self._state_id += 1
        return self.snapshot()

    @property
    def physics_dt_s(self) -> float:
        return float(self._model.opt.timestep)

    @property
    def model(self) -> mujoco.MjModel:
        """Expose immutable model metadata without exposing the owned MjData."""
        self._assert_owner()
        return self._model

    def capture_fixed_rgbd(
        self,
        builder: MujocoFixedRgbdVisualSceneBuilderV1,
        *,
        scene_id: str,
        observation_id: int,
    ) -> FixedRgbdCaptureV1:
        """Render a fresh observation while keeping all MjData access in the backend."""
        self._assert_owner()
        return builder.capture(
            self._data,
            scene_id=scene_id,
            observation_id=observation_id,
        )

    @property
    def arm_joint_limits(self) -> tuple[FloatArray, FloatArray]:
        self._assert_owner()
        ranges = self._model.jnt_range[self._arm_joint_ids]
        return ranges[:, 0].copy(), ranges[:, 1].copy()

    def reset(self) -> RawStateSnapshot:
        self._assert_owner()
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._keyframe_id)
        if self._bowl_initial_position_world_m is not None:
            self._data.qpos[
                self._bowl_qpos_address : self._bowl_qpos_address + 3
            ] = self._bowl_initial_position_world_m
            self._data.qvel[
                self._bowl_dof_address : self._bowl_dof_address + 6
            ] = 0.0
        for name, position in self._object_initial_positions_world_m.items():
            _, qpos_address, dof_address, _ = self._object_metadata[name]
            self._data.qpos[qpos_address : qpos_address + 3] = position
            self._data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._action_epoch = 0
        self._state_id += 1
        return self.snapshot()

    def snapshot(self) -> RawStateSnapshot:
        self._assert_owner()
        base_rotation_world = self._data.xmat[self._base_body_id].reshape(3, 3)
        tcp_rotation_world = self._data.site_xmat[self._tcp_site_id].reshape(3, 3)
        tcp_position_world = self._data.site_xpos[self._tcp_site_id]
        tcp_position_base = np.einsum(
            "ji,j->i",
            base_rotation_world,
            tcp_position_world - self._data.xpos[self._base_body_id],
        )
        tcp_rotation_base = np.einsum(
            "ji,jk->ik", base_rotation_world, tcp_rotation_world
        )
        tcp_quaternion_base = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(tcp_quaternion_base, tcp_rotation_base.ravel())
        bowl_velocity = self._data.cvel[self._bowl_body_id]
        bowl_position_world = self._data.xpos[self._bowl_body_id]
        bowl_position_base = np.einsum(
            "ji,j->i",
            base_rotation_world,
            bowl_position_world - self._data.xpos[self._base_body_id],
        )
        bowl_rotation_world = self._data.xmat[self._bowl_body_id].reshape(3, 3)
        bowl_rotation_base = np.einsum(
            "ji,jk->ik", base_rotation_world, bowl_rotation_world
        )
        bowl_quaternion_base = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(bowl_quaternion_base, bowl_rotation_base.ravel())
        target_position_base = (
            None if self._target_body_id is None
            else base_rotation_world.T @ (
                self._data.xpos[self._target_body_id] - self._data.xpos[self._base_body_id]
            )
        )
        return RawStateSnapshot(
            state_id=self._state_id,
            action_epoch=self._action_epoch,
            simulation_time_s=float(self._data.time),
            joint_positions_rad=tuple(
                float(value) for value in self._data.qpos[self._arm_qpos_addresses]
            ),  # type: ignore[arg-type]
            joint_velocities_rad_s=tuple(
                float(value) for value in self._data.qvel[self._arm_dof_addresses]
            ),  # type: ignore[arg-type]
            gripper_joint_m=float(self._data.qpos[self._gripper_qpos_address]),
            gripper_target_m=float(self._data.ctrl[self._gripper_actuator_id]),
            tcp_position_world_m=tuple(float(value) for value in tcp_position_world),  # type: ignore[arg-type]
            tcp_position_robot_base_m=tuple(float(value) for value in tcp_position_base),  # type: ignore[arg-type]
            tcp_orientation_robot_base_wxyz=tuple(
                float(value) for value in tcp_quaternion_base
            ),  # type: ignore[arg-type]
            bowl_position_world_m=tuple(
                float(value) for value in bowl_position_world
            ),  # type: ignore[arg-type]
            bowl_position_robot_base_m=tuple(
                float(value) for value in bowl_position_base
            ),  # type: ignore[arg-type]
            bowl_orientation_robot_base_wxyz=tuple(
                float(value) for value in bowl_quaternion_base
            ),  # type: ignore[arg-type]
            bowl_linear_velocity_world_mps=tuple(
                float(value) for value in bowl_velocity[3:]
            ),  # type: ignore[arg-type]
            bowl_angular_velocity_world_radps=tuple(
                float(value) for value in bowl_velocity[:3]
            ),  # type: ignore[arg-type]
            target_position_robot_base_m=(
                None if target_position_base is None
                else tuple(float(value) for value in target_position_base)
            ),  # type: ignore[arg-type]
            contacts=self._contact_snapshot(),
            right_gripper_joint_m=float(self._data.qpos[self._right_gripper_qpos_address]),
        )

    def object_box_corners_robot_base_m(
        self, body_name: str, snapshot: RawStateSnapshot | None = None
    ) -> FloatArray:
        """Actual collision corners for hidden evaluation only, never perception."""
        self._assert_owner()
        geom_ids = self._object_metadata[body_name][3]
        if len(geom_ids) != 1:
            raise ValueError(f"hidden box evaluation requires one collision geom: {body_name}")
        geom_id = next(iter(geom_ids))
        if self._model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"hidden box evaluation requires a box: {body_name}")
        signs = np.asarray([
            (x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
        ])
        if snapshot is not None:
            geom_rotation = np.empty(9)
            mujoco.mju_quat2Mat(geom_rotation, self._model.geom_quat[geom_id])
            body_rotation = np.empty(9)
            mujoco.mju_quat2Mat(
                body_rotation, np.asarray(snapshot.bowl_orientation_robot_base_wxyz)
            )
            local = (
                (signs * self._model.geom_size[geom_id]) @ geom_rotation.reshape(3, 3).T
                + self._model.geom_pos[geom_id]
            )
            return (
                local @ body_rotation.reshape(3, 3).T
                + np.asarray(snapshot.bowl_position_robot_base_m)
            )
        world = (
            (signs * self._model.geom_size[geom_id])
            @ self._data.geom_xmat[geom_id].reshape(3, 3).T
            + self._data.geom_xpos[geom_id]
        )
        return (
            (world - self._data.xpos[self._base_body_id])
            @ self._data.xmat[self._base_body_id].reshape(3, 3)
        )

    def _contact_snapshot(self) -> PhysicalContactSnapshot:
        left_force = 0.0
        right_force = 0.0
        table_force = 0.0
        for contact_index in range(self._data.ncon):
            contact = self._data.contact[contact_index]
            pair = frozenset((int(contact.geom1), int(contact.geom2)))
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self._model, self._data, contact_index, force)
            normal_force = abs(float(force[0]))
            if self._left_finger_geom_id in pair and pair & self._bowl_geom_ids:
                left_force += normal_force
            if pair & self._additional_effector_geom_ids and pair & self._bowl_geom_ids:
                left_force += normal_force
            if self._right_finger_geom_id in pair and pair & self._bowl_geom_ids:
                right_force += normal_force
            support_geoms = (
                {self._table_geom_id}
                | (self._all_object_geom_ids - self._bowl_geom_ids)
                | self._additional_support_geom_ids
            )
            if pair & self._bowl_geom_ids and pair & support_geoms:
                table_force += normal_force
        return PhysicalContactSnapshot(
            left_finger_bowl_contact=left_force > 0.0,
            right_finger_bowl_contact=right_force > 0.0,
            left_finger_bowl_normal_force_n=left_force,
            right_finger_bowl_normal_force_n=right_force,
            bowl_table_contact=table_force > 0.0,
            bowl_table_normal_force_n=table_force,
        )

    def arm_jacobian_world(self) -> tuple[FloatArray, FloatArray]:
        self._assert_owner()
        jacobian_position = np.zeros((3, self._model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self._model,
            self._data,
            jacobian_position,
            jacobian_rotation,
            self._tcp_site_id,
        )
        return (
            jacobian_position[:, self._arm_dof_addresses].copy(),
            jacobian_rotation[:, self._arm_dof_addresses].copy(),
        )

    def base_rotation_world(self) -> FloatArray:
        self._assert_owner()
        return self._data.xmat[self._base_body_id].reshape(3, 3).copy()

    def tcp_rotation_world(self) -> FloatArray:
        self._assert_owner()
        return self._data.site_xmat[self._tcp_site_id].reshape(3, 3).copy()

    def set_arm_compensated_position_targets(self, nominal_targets_rad: FloatArray) -> None:
        """Track nominal joint positions with MuJoCo bias-force feed-forward."""
        self._assert_owner()
        nominal_targets = np.asarray(nominal_targets_rad, dtype=np.float64)
        if nominal_targets.shape != (6,):
            raise ValueError(f"Expected 6 arm targets, got shape {nominal_targets.shape}")
        lower, upper = self.arm_joint_limits
        if np.any(nominal_targets < lower) or np.any(nominal_targets > upper):
            raise ValueError("Nominal arm target exceeds MuJoCo joint limits")
        compensation = (
            self._data.qfrc_bias[self._arm_dof_addresses]
            / self._arm_actuator_gears
            / self._arm_position_gains
        )
        compensated_targets = nominal_targets + compensation
        control_ranges = self._model.actuator_ctrlrange[self._arm_actuator_ids]
        if np.any(compensated_targets < control_ranges[:, 0]) or np.any(
            compensated_targets > control_ranges[:, 1]
        ):
            raise ValueError("Bias-compensated arm target exceeds actuator control limits")
        self._data.ctrl[self._arm_actuator_ids] = compensated_targets

    def set_gripper_position_target(self, target_m: float) -> None:
        self._assert_owner()
        control_range = self._model.actuator_ctrlrange[self._gripper_actuator_id]
        if target_m < control_range[0] or target_m > control_range[1]:
            raise ValueError(f"Gripper target {target_m} exceeds {control_range.tolist()}")
        self._data.ctrl[self._gripper_actuator_id] = target_m

    def step(self, count: int = 1) -> RawStateSnapshot:
        self._assert_owner()
        if count <= 0:
            raise ValueError("step count must be positive")
        for _ in range(count):
            mujoco.mj_step(self._model, self._data)
            self._capture_frame()
        self._state_id += 1
        return self.snapshot()

    def commit_action(self) -> RawStateSnapshot:
        self._assert_owner()
        self._action_epoch += 1
        self._state_id += 1
        return self.snapshot()

    def _capture_frame(self, force: bool = False) -> None:
        if self._renderer is None or self._frame_sink is None:
            return
        if not force and self._data.time + 1.0e-12 < self._next_frame_time_s:
            return
        self._renderer.update_scene(self._data, camera="debug_camera")
        self._frame_sink(self._renderer.render())
        self._next_frame_time_s = float(self._data.time) + self._video_period_s

    def close(self) -> None:
        self._assert_owner()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
