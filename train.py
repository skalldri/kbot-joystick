"""Defines simple task for training a joystick walking policy for K-Bot."""

import asyncio
import functools
import math
from dataclasses import dataclass
from typing import Self

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from step import step_fn
from model import Actor, Critic, Model, ZEROS


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=512,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=2,
        help="The depth for the MLPs",
    )
    num_hidden_layers: int = xax.field(
        value=2,
        help="The number of hidden layers for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=10,
        help="The number of mixtures for the actor.",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )
    warmup_steps: int = xax.field(
        value=100,
        help="The number of steps to warm up the learning rate.",
    )
    grad_clip: float = xax.field(
        value=2.0,
        help="Gradient clip for the Adam optimizer.",
    )


# TODO put this in xax?
def rotate_quat_by_quat(
    quat_to_rotate: Array,
    rotating_quat: Array,
    inverse: bool = False,
    eps: float = 1e-6,
) -> Array:
    """Rotates one quaternion by another quaternion through quaternion multiplication.

    This performs the operation: rotating_quat * quat_to_rotate * rotating_quat^(-1) if inverse=False
    or rotating_quat^(-1) * quat_to_rotate * rotating_quat if inverse=True

    Args:
        quat_to_rotate: The quaternion being rotated (w,x,y,z), shape (*, 4)
        rotating_quat: The quaternion performing the rotation (w,x,y,z), shape (*, 4)
        inverse: If True, rotate by the inverse of rotating_quat
        eps: Small epsilon value to avoid division by zero in normalization

    Returns:
        The rotated quaternion (w,x,y,z), shape (*, 4)
    """
    # Normalize both quaternions
    quat_to_rotate = quat_to_rotate / (
        jnp.linalg.norm(quat_to_rotate, axis=-1, keepdims=True) + eps
    )
    rotating_quat = rotating_quat / (
        jnp.linalg.norm(rotating_quat, axis=-1, keepdims=True) + eps
    )

    # If inverse requested, conjugate the rotating quaternion (negate x,y,z components)
    if inverse:
        rotating_quat = rotating_quat.at[..., 1:].multiply(-1)

    # Extract components of both quaternions
    w1, x1, y1, z1 = jnp.split(rotating_quat, 4, axis=-1)  # rotating quaternion
    w2, x2, y2, z2 = jnp.split(quat_to_rotate, 4, axis=-1)  # quaternion being rotated

    # Quaternion multiplication formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = jnp.concatenate([w, x, y, z], axis=-1)

    # Normalize result
    return result / (jnp.linalg.norm(result, axis=-1, keepdims=True) + eps)


def _is_zero_command(
    command: Array, command_idx: int, threshold: float = 1e-3
) -> Array:
    """Check if the command up-to the given index is close to zero."""
    return jnp.linalg.norm(command[:, : command_idx + 1], axis=-1) < threshold


@attrs.define(frozen=True, kw_only=True)
class ContactForcePenalty(ksim.Reward):
    """Penalises vertical forces above threshold."""

    scale: float = -1.0
    max_contact_force: float = 350.0
    sensor_names: tuple[str, ...]

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        forces = jnp.stack([traj.obs[n] for n in self.sensor_names], axis=-1)
        cost = jnp.clip(jnp.abs(forces[:, 2, :]) - self.max_contact_force, 0)
        return jnp.sum(cost, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class SimpleSingleFootContactReward(ksim.Reward):
    """Reward having one and only one foot in contact with the ground, while walking."""

    scale: float = 1.0

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        left_contact = jnp.where(
            traj.obs["sensor_observation_left_foot_touch"] > 0.1, True, False
        ).squeeze()
        right_contact = jnp.where(
            traj.obs["sensor_observation_right_foot_touch"] > 0.1, True, False
        ).squeeze()
        single = jnp.logical_xor(left_contact, right_contact).squeeze()

        is_zero_cmd = _is_zero_command(
            command=traj.command["unified_command"],
            command_idx=UnifiedCommand.MOVEMENT_COMMAND_END_IDX,
        )

        reward = jnp.where(is_zero_cmd, 1.0, single)
        return reward


@attrs.define(frozen=True, kw_only=True)
class SingleFootContactReward(ksim.StatefulReward):
    """Reward having one and only one foot in contact with the ground, while walking.

    Allows for small grace period when both feet are in contact for less jumpy gaits.
    """

    scale: float = 1.0
    ctrl_dt: float = 0.02
    grace_period: float = 0.2  # seconds

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        return jnp.array([0.0])

    def get_reward_stateful(
        self, traj: ksim.Trajectory, reward_carry: PyTree
    ) -> tuple[Array, PyTree]:
        left_contact = jnp.where(
            traj.obs["sensor_observation_left_foot_touch"] > 0.1, True, False
        ).squeeze()
        right_contact = jnp.where(
            traj.obs["sensor_observation_right_foot_touch"] > 0.1, True, False
        ).squeeze()
        single = jnp.logical_xor(left_contact, right_contact)

        def _body(
            time_since_single_contact: Array, is_single_contact: Array
        ) -> tuple[Array, Array]:
            new_time = jnp.where(
                is_single_contact, 0.0, time_since_single_contact + self.ctrl_dt
            )
            return new_time, new_time

        carry, time_since_single_contact = jax.lax.scan(_body, reward_carry, single)
        single_contact_grace = time_since_single_contact < self.grace_period
        is_zero_cmd = _is_zero_command(
            command=traj.command["unified_command"],
            command_idx=UnifiedCommand.MOVEMENT_COMMAND_END_IDX,
        )
        reward = jnp.where(is_zero_cmd, 0.0, single_contact_grace.squeeze())
        return reward, carry


@attrs.define(frozen=True, kw_only=True)
class FeetAirtimeReward(ksim.StatefulReward):
    """Encourages reasonable step frequency by rewarding long swing phases and penalizing quick stepping."""

    scale: float = 1.0
    ctrl_dt: float = 0.02
    touchdown_penalty: float = 0.4
    scale_by_curriculum: bool = False

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        # initial left and right airtime
        return jnp.array([0.0, 0.0])

    def _airtime_sequence(
        self, initial_airtime: Array, contact_bool: Array, done: Array
    ) -> tuple[Array, Array]:
        """Returns an array with the airtime (in seconds) for each timestep."""

        def _body(time_since_liftoff: Array, is_contact: Array) -> tuple[Array, Array]:
            new_time = jnp.where(is_contact, 0.0, time_since_liftoff + self.ctrl_dt)
            return new_time, new_time

        # or with done to reset the airtime counter when the episode is done
        contact_or_done = jnp.logical_or(contact_bool, done)
        carry, airtime = jax.lax.scan(_body, initial_airtime, contact_or_done)
        return carry, airtime

    def get_reward_stateful(
        self, traj: ksim.Trajectory, reward_carry: PyTree
    ) -> tuple[Array, PyTree]:
        left_contact = jnp.where(
            traj.obs["sensor_observation_left_foot_touch"] > 0.1, True, False
        )[
            :, 0
        ]  # .squeeze()
        right_contact = jnp.where(
            traj.obs["sensor_observation_right_foot_touch"] > 0.1, True, False
        )[
            :, 0
        ]  # .squeeze()

        # airtime counters
        left_carry, left_air = self._airtime_sequence(
            reward_carry[0], left_contact, traj.done
        )
        right_carry, right_air = self._airtime_sequence(
            reward_carry[1], right_contact, traj.done
        )

        reward_carry = jnp.array([left_carry, right_carry])

        # touchdown boolean (0→1 transition)
        def touchdown(c: Array) -> Array:
            prev = jnp.concatenate([jnp.array([False]), c[:-1]])
            return jnp.logical_and(c, jnp.logical_not(prev))

        td_l = touchdown(left_contact)
        td_r = touchdown(right_contact)

        left_air_shifted = jnp.roll(left_air, 1)
        right_air_shifted = jnp.roll(right_air, 1)

        left_feet_airtime_reward = (
            left_air_shifted - self.touchdown_penalty
        ) * td_l.astype(jnp.float32)
        right_feet_airtime_reward = (
            right_air_shifted - self.touchdown_penalty
        ) * td_r.astype(jnp.float32)

        reward = left_feet_airtime_reward + right_feet_airtime_reward

        # standing mask
        is_zero_cmd = _is_zero_command(
            command=traj.command["unified_command"],
            command_idx=UnifiedCommand.MOVEMENT_COMMAND_END_IDX,
        )
        reward = jnp.where(is_zero_cmd, 0.0, reward)

        return reward, reward_carry


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        zeros = {k: v for k, v, _ in ZEROS}
        weights = {k: v for k, _, v in ZEROS}
        joint_targets = [zeros[name] for name in names]
        joint_weights = [weights[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            joint_weights=tuple(joint_weights),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


# TODO implement this as a reward with kernel
@attrs.define(frozen=True, kw_only=True)
class BentArmPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_right_shoulder_pitch_03",
                "dof_right_shoulder_roll_03",
                "dof_right_shoulder_yaw_02",
                "dof_right_elbow_02",
                "dof_right_wrist_00",
                "dof_left_shoulder_pitch_03",
                "dof_left_shoulder_roll_03",
                "dof_left_shoulder_yaw_02",
                "dof_left_elbow_02",
                "dof_left_wrist_00",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the linear velocity."""

    error_scale: float = attrs.field(default=0.25)
    linvel_obs_name: str = attrs.field(default="sensor_observation_base_site_linvel")
    command_name: str = attrs.field(default="unified_command")
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # need to get lin vel obs from sensor, because xvel is not available in Trajectory.
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(
                f"Observation {self.linvel_obs_name} not found; add it as an observation in your task."
            )

        # Get global frame velocities
        global_vel = trajectory.obs[self.linvel_obs_name]

        # get base quat, only yaw.
        # careful to only rotate in z, disregard rx and ry, bad conflict with roll and pitch.
        base_euler = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        base_euler = base_euler.at[:, :2].set(0.0)
        base_z_quat = xax.euler_to_quat(base_euler)

        # rotate local frame commands to global frame
        robot_vel_cmd = (
            jnp.zeros_like(global_vel)
            .at[:, :2]
            .set(trajectory.command[self.command_name][:, :2])
        )
        global_vel_cmd = xax.rotate_vector_by_quat(
            robot_vel_cmd, base_z_quat, inverse=False
        )

        # drop vz. vz conflicts with base height reward.
        global_vel_xy_cmd = global_vel_cmd[:, :2]
        global_vel_xy = global_vel[:, :2]

        # now compute error. special trick: different kernels for standing and walking.
        zero_cmd_mask = _is_zero_command(
            command=trajectory.command[self.command_name],
            command_idx=UnifiedCommand.MOVEMENT_COMMAND_END_IDX,
        )

        vel_error = jnp.linalg.norm(global_vel_xy - global_vel_xy_cmd, axis=-1)
        error = jnp.where(zero_cmd_mask, vel_error, 2 * jnp.square(vel_error))
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(ksim.Reward):
    """Reward for tracking the heading using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")
    angvel_obs_name: str = attrs.field(default="sensor_observation_base_site_angvel")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        if self.angvel_obs_name not in trajectory.obs:
            raise ValueError(
                f"Observation {self.angvel_obs_name} not found; add it as an observation in your task."
            )

        global_angvel = trajectory.obs[self.angvel_obs_name]

        # get only the z component of the angular velocity
        global_angvel_z = global_angvel[:, 2]
        angvel_z_command = trajectory.command[self.command_name][
            :, UnifiedCommand.ANGULAR_VELOCITY_Z_COMMAND_IDX
        ]

        angvel_error = jnp.linalg.norm(global_angvel_z - angvel_z_command, axis=-1)

        # now compute error. special trick: different kernels for standing and walking.
        zero_cmd_mask = _is_zero_command(
            command=trajectory.command[self.command_name],
            command_idx=UnifiedCommand.MOVEMENT_COMMAND_END_IDX,
        )

        error = jnp.where(zero_cmd_mask, angvel_error, 2 * jnp.square(angvel_error))
        return jnp.exp(-error / self.error_scale)


@attrs.define(frozen=True)
class XYOrientationReward(ksim.Reward):
    """Reward for tracking the xy base orientation using quaternion-based error computation."""

    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="unified_command")

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        euler_orientation = xax.quat_to_euler(trajectory.xquat[:, 1, :])
        euler_orientation = euler_orientation.at[:, 2].set(0.0)  # ignore yaw
        base_xy_quat = xax.euler_to_quat(euler_orientation)

        commanded_euler = jnp.stack(
            [
                trajectory.command[self.command_name][:, 9],
                trajectory.command[self.command_name][:, 10],
                jnp.zeros_like(trajectory.command[self.command_name][:, 10]),
            ],
            axis=-1,
        )
        base_xy_quat_cmd = xax.euler_to_quat(commanded_euler)

        quat_error = 1 - jnp.sum(base_xy_quat_cmd * base_xy_quat, axis=-1) ** 2
        return jnp.exp(-quat_error / self.error_scale)


@attrs.define(frozen=True)
class BaseHeightReward(ksim.Reward):
    """Reward for keeping the base height at the commanded height."""

    error_scale: float = attrs.field(default=0.25)
    standard_height: float = attrs.field(default=1.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        current_height = trajectory.xpos[
            :, 1, 2
        ]  # 1st body, because world is 0. 2nd element is z.
        commanded_height = (
            trajectory.command["unified_command"][
                :, UnifiedCommand.BODY_HEIGHT_COMMAND_IDX
            ]
            + self.standard_height
        )

        height_error = jnp.abs(current_height - commanded_height)
        is_zero_cmd = _is_zero_command(
            command=trajectory.command["unified_command"],
            command_idx=UnifiedCommand.MOVEMENT_COMMAND_END_IDX,
        )
        height_error = jnp.where(
            is_zero_cmd, height_error, height_error**2
        )  # smooth kernel for walking.
        return jnp.exp(-height_error / self.error_scale)


@attrs.define(frozen=True)
class FeetPositionReward(ksim.Reward):
    """Reward for keeping the feet next to each other when standing still."""

    error_scale: float = attrs.field(default=0.25)
    stance_width: float = attrs.field(default=0.3)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        feet_pos = trajectory.obs["feet_position_observation"]
        left_foot_pos = feet_pos[:, :3]
        right_foot_pos = feet_pos[:, 3:]

        # Calculate stance errors
        stance_x_error = jnp.abs(left_foot_pos[:, 0] - right_foot_pos[:, 0])
        stance_y_error = jnp.abs(
            jnp.abs(left_foot_pos[:, 1] - right_foot_pos[:, 1]) - self.stance_width
        )
        stance_error = stance_x_error + stance_y_error
        reward = jnp.exp(-stance_error / self.error_scale)

        # standing?
        zero_cmd_mask = _is_zero_command(
            command=trajectory.command["unified_command"],
            command_idx=UnifiedCommand.MOVEMENT_COMMAND_END_IDX,
        )
        reward = jnp.where(zero_cmd_mask, reward, 0.0)
        return reward


@attrs.define(frozen=True)
class FeetOrientationReward(ksim.Reward):
    """Reward for keeping feet pitch and roll oriented parallel to the ground."""

    scale: float = attrs.field(default=1.0)
    error_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        left_foot_euler = xax.quat_to_euler(trajectory.xquat[:, 23, :])
        right_foot_euler = xax.quat_to_euler(trajectory.xquat[:, 18, :])

        straight_foot_euler = jnp.stack([-jnp.pi / 2, 0], axis=-1)  # ignore yaw

        left_error = jnp.abs(left_foot_euler[:, :2] - straight_foot_euler).sum(axis=-1)
        right_error = jnp.abs(right_foot_euler[:, :2] - straight_foot_euler).sum(
            axis=-1
        )

        total_error = left_error + right_error
        return jnp.exp(-total_error / self.error_scale)


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    foot_left_idx: int
    foot_right_idx: int
    floor_threshold: float = 0.0
    in_robot_frame: bool = True

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_site_name: str,
        foot_right_site_name: str,
        floor_threshold: float = 0.0,
        in_robot_frame: bool = True,
    ) -> Self:
        fl = ksim.get_site_data_idx_from_name(physics_model, foot_left_site_name)
        fr = ksim.get_site_data_idx_from_name(physics_model, foot_right_site_name)
        return cls(
            foot_left_idx=fl,
            foot_right_idx=fr,
            floor_threshold=floor_threshold,
            in_robot_frame=in_robot_frame,
        )

    def observe(
        self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        fl_ndarray = ksim.get_site_pose(state.physics_state.data, self.foot_left_idx)[
            0
        ] + jnp.array([0.0, 0.0, self.floor_threshold])
        fr_ndarray = ksim.get_site_pose(state.physics_state.data, self.foot_right_idx)[
            0
        ] + jnp.array([0.0, 0.0, self.floor_threshold])

        if self.in_robot_frame:
            # Transform foot positions to robot frame
            base_quat = state.physics_state.data.qpos[3:7]  # Base quaternion
            fl = xax.rotate_vector_by_quat(
                jnp.array(fl_ndarray), base_quat, inverse=True
            )
            fr = xax.rotate_vector_by_quat(
                jnp.array(fr_ndarray), base_quat, inverse=True
            )

        return jnp.concatenate([fl, fr], axis=-1)


@attrs.define(frozen=True)
class BaseHeightObservation(ksim.Observation):
    """Observation of the base height."""

    def observe(
        self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        return state.physics_state.data.xpos[1, 2:]


@attrs.define(kw_only=True)
class UnifiedLinearVelocityCommandMarker(ksim.vis.Marker):
    """Visualise the planar (x,y) velocity command from unified command."""

    command_name: str = attrs.field()
    size: float = attrs.field(default=0.03)
    arrow_scale: float = attrs.field(default=0.1)
    height: float = attrs.field(default=0.5)
    base_length: float = attrs.field(default=0.25)

    def update(self, trajectory: ksim.Trajectory) -> None:
        cmd = trajectory.command[self.command_name]
        vx, vy = float(cmd[0]), float(cmd[1])
        speed = (vx * vx + vy * vy) ** 0.5
        self.pos = (0.0, 0.0, self.height)

        # Always show arrow with base length plus scaling
        self.geom = mujoco.mjtGeom.mjGEOM_ARROW
        arrow_length = self.base_length + self.arrow_scale * speed
        self.scale = (self.size, self.size, arrow_length)

        if speed < 1e-4:  # zero command → point forward, grey color
            self.orientation = self.quat_from_direction((1.0, 0.0, 0.0))
            self.rgba = (0.8, 0.8, 0.8, 0.8)
        else:  # non-zero command → point in command direction, green color
            self.orientation = self.quat_from_direction((vx, vy, 0.0))
            self.rgba = (0.2, 0.8, 0.2, 0.8)

    @classmethod
    def get(
        cls,
        command_name: str,
        *,
        arrow_scale: float = 0.1,
        height: float = 0.5,
        base_length: float = 0.25,
    ) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_ARROW,
            scale=(0.03, 0.03, base_length),
            arrow_scale=arrow_scale,
            height=height,
            base_length=base_length,
            track_rotation=True,
        )


@attrs.define(kw_only=True)
class UnifiedAbsoluteYawCommandMarker(ksim.vis.Marker):
    """Visualise the absolute yaw command from unified command."""

    command_name: str = attrs.field()
    size: float = attrs.field(default=0.02)
    height: float = attrs.field(default=0.7)
    arrow_scale: float = attrs.field(default=0.1)
    base_length: float = attrs.field(default=0.25)

    def update(self, trajectory: ksim.Trajectory) -> None:
        cmd = trajectory.command[self.command_name]
        yaw = float(cmd[3])  # yaw command is in position 3
        self.pos = (0.0, 0.0, self.height)

        # Always show arrow with base length plus scaling
        self.geom = mujoco.mjtGeom.mjGEOM_ARROW
        arrow_length = self.base_length + self.arrow_scale * abs(yaw)
        self.scale = (self.size, self.size, arrow_length)

        if abs(yaw) < 1e-4:  # zero command → point forward, grey color
            self.orientation = self.quat_from_direction((1.0, 0.0, 0.0))
            self.rgba = (0.8, 0.8, 0.8, 0.8)
        else:  # non-zero command → point in yaw direction, blue color
            # Convert yaw to direction vector (rotate around z-axis)
            direction_x = jnp.cos(yaw)
            direction_y = jnp.sin(yaw)
            self.orientation = self.quat_from_direction(
                (float(direction_x), float(direction_y), 0.0)
            )
            self.rgba = (0.2, 0.2, 0.8, 0.8)

    @classmethod
    def get(
        cls,
        command_name: str,
        *,
        arrow_scale: float = 0.1,
        height: float = 0.7,
        base_length: float = 0.25,
    ) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_ARROW,
            scale=(0.02, 0.02, base_length),
            arrow_scale=arrow_scale,
            height=height,
            base_length=base_length,
            track_rotation=False,
        )


@attrs.define(frozen=True)
class UnifiedCommand(ksim.Command):
    """Unifiying all commands into one to allow for covariance control.

    UnifiedCommand return an array consisting of:
    [
        vx, = Velocity in x direction
        vy, = Velocity in y direction
        wz, = Angular velocity around z-axis
        bh, = Body height
        rx, = Rotation around x-axis
        ry, = Rotation around y-axis
    ]
    """

    vx_range: tuple[float, float] = attrs.field()
    vy_range: tuple[float, float] = attrs.field()
    wz_range: tuple[float, float] = attrs.field()
    bh_range: tuple[float, float] = attrs.field()
    bh_standing_range: tuple[float, float] = attrs.field()
    rx_range: tuple[float, float] = attrs.field()
    ry_range: tuple[float, float] = attrs.field()
    ctrl_dt: float = attrs.field()
    switch_prob: float = attrs.field()

    VELOCITY_X_COMMAND_IDX = 0
    VELOCITY_Y_COMMAND_IDX = 1
    ANGULAR_VELOCITY_Z_COMMAND_IDX = 2

    MOVEMENT_COMMAND_END_IDX = 2

    BODY_HEIGHT_COMMAND_IDX = 3
    BODY_FRONT_BACK_PITCH_COMMAND_IDX = 4
    BODY_LEFT_RIGHT_ROLL_COMMAND_IDX = 5

    NUM_COMMANDS = 6

    def initial_command(
        self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        rng_a, rng_b, rng_c, rng_d, rng_e, rng_f, rng_g, rng_h = jax.random.split(
            rng, 8
        )

        # cmd  = [vx, vy, wz, bh, rx, ry]
        vx = jax.random.uniform(
            rng_b, (1,), minval=self.vx_range[0], maxval=self.vx_range[1]
        )
        vy = jax.random.uniform(
            rng_c, (1,), minval=self.vy_range[0], maxval=self.vy_range[1]
        )
        wz = jax.random.uniform(
            rng_d, (1,), minval=self.wz_range[0], maxval=self.wz_range[1]
        )
        bh = jax.random.uniform(
            rng_e, (1,), minval=self.bh_range[0], maxval=self.bh_range[1]
        )
        bhs = jax.random.uniform(
            rng_f,
            (1,),
            minval=self.bh_standing_range[0],
            maxval=self.bh_standing_range[1],
        )
        rx = jax.random.uniform(
            rng_g, (1,), minval=self.rx_range[0], maxval=self.rx_range[1]
        )
        ry = jax.random.uniform(
            rng_h, (1,), minval=self.ry_range[0], maxval=self.ry_range[1]
        )

        # don't like super small velocity commands
        vx = jnp.where(jnp.abs(vx) < 0.09, 0.0, vx)
        vy = jnp.where(jnp.abs(vy) < 0.09, 0.0, vy)
        wz = jnp.where(jnp.abs(wz) < 0.09, 0.0, wz)

        _ = jnp.zeros_like(vx)

        # Create each mode's command vector
        forward_cmd = jnp.concatenate([vx, _, _, bh, _, _])
        sideways_cmd = jnp.concatenate([_, vy, _, bh, _, _])
        rotate_cmd = jnp.concatenate([_, _, wz, bh, _, _])
        omni_cmd = jnp.concatenate([vx, vy, wz, bh, _, _])
        stand_cmd = jnp.concatenate([_, _, _, bhs, rx, ry])

        mode = jax.random.randint(rng_a, (), minval=0, maxval=5)
        # Use JAX's where() to select the appropriate command
        cmd = jnp.where(
            mode == 0,
            forward_cmd,
            jnp.where(
                mode == 1,
                sideways_cmd,
                jnp.where(
                    mode == 2, rotate_cmd, jnp.where(mode == 3, omni_cmd, stand_cmd)
                ),
            ),
        )

        return cmd

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)

        # Generate a new_command that we _might_ switch to commanding
        new_command = self.initial_command(physics_data, curriculum_level, rng_b)

        # Generate a boolean to decide whether to switch commands.
        # switch_prob controls the likelihood of producing a "True" result from the bernoulli distribution.
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)

        # If switch_mask is True, use new_command, otherwise keep prev_command
        return jnp.where(switch_mask, new_command, prev_command)

    def get_markers(self) -> list[ksim.vis.Marker]:
        """Return markers for visualizing the unified command components."""
        return [
            UnifiedAbsoluteYawCommandMarker.get(
                command_name=self.command_name,
                height=0.7,
            ),
            UnifiedLinearVelocityCommandMarker.get(
                command_name=self.command_name,
                height=0.5,
            ),
        ]


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        scheduler = optax.warmup_constant_schedule(
            init_value=self.config.learning_rate * 0.01,
            peak_value=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
        )

        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.add_decayed_weights(self.config.adam_weight_decay),
            optax.scale_by_adam(),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(
            ksim.get_mujoco_model_path("kbot-headless", name="robot")
        )
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot-headless"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata is not available")
        return metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(
        self, physics_model: ksim.PhysicsModel
    ) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(
                scale_lower=0.90, scale_upper=1.10
            ),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(
                scale_lower=math.radians(-2), scale_upper=math.radians(2)
            ),
            ksim.FloorFrictionRandomizer.from_geom_name(
                model=physics_model,
                floor_geom_name="floor",
                scale_lower=0.1,
                scale_upper=2.0,
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.LinearPushEvent(
                linvel=1.0,
                interval_range=(2.0, 5.0),
            ),
            ksim.JumpEvent(
                jump_height_range=(0.1, 0.5),
                interval_range=(2.0, 5.0),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(
                physics_model, {k: v for k, v, _ in ZEROS}, scale=0.1
            ),
            ksim.RandomJointVelocityReset(),
            ksim.RandomHeightReset(),
        ]

    def get_observations(
        self, physics_model: ksim.PhysicsModel
    ) -> list[ksim.Observation]:
        return [
            # ACTOR OBSERVATIONS
            # ##################
            # These observations correspond with the ones available
            # in the kinfer runtime. These are the only observations allowed to be used
            # by the actor.
            ksim.JointPositionObservation(noise=math.radians(2)),  # joint_angles
            ksim.JointVelocityObservation(
                noise=math.radians(10)
            ),  # joint_angular_velocities
            # TODO: create an observation for initial_heading
            # TBH I don't think having an "initial_heading" observation is useful in the long run
            ksim.ProjectedGravityObservation.create(  # projected_gravity
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                noise=math.radians(1),
            ),
            ksim.SensorObservation.create(  # accelerometer
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=0.001,  # Add 0.001 m/s^2 of gaussian noise to the IMU
            ),
            ksim.SensorObservation.create(  # gyroscope
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(10),  # Add 10 degrees of gaussian noise to the IMU
            ),
            # TODO: time observation?
            # ##################
            # CRITIC OBSERVATIONS
            # ###################
            # These additional observations are available to be used by the critic model.
            # They _must_ not be used by the actor: the kinfer runtime does not expose these to the actor model.
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.SensorObservation.create(
                physics_model=physics_model, sensor_name="left_foot_touch", noise=0.0
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model, sensor_name="right_foot_touch", noise=0.0
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model, sensor_name="base_site_linvel", noise=0.0
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model, sensor_name="base_site_angvel", noise=0.0
            ),
            FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_site_name="left_foot",
                foot_right_site_name="right_foot",
                floor_threshold=0.0,
                in_robot_frame=True,
            ),
            BaseHeightObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            UnifiedCommand(
                vx_range=(-0.5, 2.0),  # m/s
                vy_range=(-0.5, 0.5),  # m/s
                wz_range=(-0.5, 0.5),  # rad/s
                # bh_range=(-0.05, 0.05), # m
                # bh_standing_range=(-0.2, 0.1), # m
                bh_range=(
                    0.0,
                    0.0,
                ),  # m # disabled for now, does not work on this robot. reward conflicts
                bh_standing_range=(0.0, 0.0),  # m
                # rx_range=(-0.3, 0.3),  # rad
                # ry_range=(-0.3, 0.3),  # rad
                # Initial training run, don't train for roll and pitch control
                rx_range=(0.0, 0.0),  # rad
                ry_range=(0.0, 0.0),  # rad
                ctrl_dt=self.config.ctrl_dt,
                switch_prob=self.config.ctrl_dt / 5,  # once per x seconds
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # cmd
            LinearVelocityTrackingReward(scale=0.3, error_scale=0.1),
            AngularVelocityTrackingReward(scale=0.1, error_scale=0.005),
            XYOrientationReward(scale=0.2, error_scale=0.03),
            BaseHeightReward(scale=0.1, error_scale=0.05, standard_height=1.0),
            # shaping
            SimpleSingleFootContactReward(scale=0.1),
            # SingleFootContactReward(scale=0.1, ctrl_dt=self.config.ctrl_dt, grace_period=0.2),
            FeetAirtimeReward(
                scale=1.0, ctrl_dt=self.config.ctrl_dt, touchdown_penalty=0.4
            ),
            FeetOrientationReward(scale=0.1, error_scale=0.25),
            BentArmPenalty.create_penalty(physics_model, scale=-0.02),
            # FeetPositionReward(scale=0.1, error_scale=0.05, stance_width=0.3),
            # sim2real
            # ksim.CtrlPenalty(scale=-0.00001),
            # ksim.ActionAccelerationPenalty(scale=-0.02, scale_by_curriculum=False),
            # ksim.JointAccelerationPenalty(scale=-0.01, scale_by_curriculum=False),
            # ksim.JointJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.LinkAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.LinkJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            # BUG: wrong sensors
            # ContactForcePenalty( # NOTE this could actually be good but eliminate until needed
            #     scale=-0.03,
            #     sensor_names=("sensor_observation_left_foot_force", "sensor_observation_right_foot_force"),
            # ),
        ]

    def get_terminations(
        self, physics_model: ksim.PhysicsModel
    ) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.6, unhealthy_z_upper=1.2),
            ksim.NotUprightTermination(max_radians=math.radians(60)),
            ksim.EpisodeLengthTermination(max_length_sec=24),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.LinearCurriculum(
            step_size=1,
            step_every_n_epochs=1,
            min_level=1.0,  # disable curriculum
        )

    def get_model(self, params: ksim.InitParams) -> Model:
        num_joints = len(ZEROS)

        num_commands = (
            2  # linear velocity command (vx, vy)
            + 1  # yaw velocity command (wz)
            + 1  # base height command
            + 2  # base roll and pitch (rx, ry)
        )  # Total: 6 commands

        num_actor_inputs = (
            num_joints * 2  # joint pos and vel
            + 3  # imu_gyro
            + 3  # imu_acc
            + num_commands
        )

        num_critic_inputs = (
            num_joints * 2  # joint pos and vel
            + 3  # imu_gyro
            + 3  # imu_acc
            + num_commands
            + 2  # feet touch (left, right)
            + 6  # feet position (xyz, left and right foot)
            + 3  # base pos (xyz)
            + 4  # base quat (wxyz)
            + 138  # COM inertia #TODO @salldritt where does this come from?
            + 230  # COM velocity #TODO @salldritt where does this come from?
            + 3  # base linear vel (xyz)
            + 3  # base angular vel (xyz)
            + num_joints  # actuator force
            + 1  # base height
        )

        return Model(
            key=params.key,
            num_actor_inputs=num_actor_inputs,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=num_critic_inputs,
            min_std=0.01,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
            num_hidden_layers=self.config.num_hidden_layers,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[xax.Distribution, Array]:
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        projected_gravity_3 = observations["projected_gravity_observation"]
        cmd = commands["unified_command"]

        # Match observations to the expected function parameter names for a standardized model function
        args = {
            "joint_angles": joint_pos_n,
            "joint_angular_velocities": joint_vel_n,
            "initial_heading": jnp.zeros_like(imu_gyro_3[..., :1]),  # Placeholder
            "projected_gravity": projected_gravity_3,
            "gyroscope": imu_gyro_3,
            "accelerometer": imu_acc_3,
            "command": cmd,
            "time": jnp.zeros_like(imu_gyro_3[..., :1]),  # Placeholder
        }

        # Mimick the behavior of the kinfer runtime provider, which will match observations to
        # input parameter names of the model function.
        action, carry = step_fn(actor=model, carry=carry, **args)
        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        # Actor observations
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        # projected_gravity_3 = observations["projected_gravity_observation"]
        cmd = commands["unified_command"]

        # privileged obs
        left_touch = observations["sensor_observation_left_foot_touch"]
        right_touch = observations["sensor_observation_right_foot_touch"]
        feet_position_6 = observations["feet_position_observation"]
        base_position_3 = observations["base_position_observation"]
        base_orientation_4 = observations["base_orientation_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        base_lin_vel_3 = observations["base_linear_velocity_observation"]
        base_ang_vel_3 = observations["base_angular_velocity_observation"]
        actuator_force_n = observations["actuator_force_observation"]
        base_height = observations["base_height_observation"]

        obs_n = jnp.concatenate(
            [
                # actor obs:
                joint_pos_n,
                joint_vel_n,  # TODO why was this / 10.0?
                imu_gyro_3,  # rad/s
                imu_acc_3,  # m/s^2
                cmd,  # 6: [vx, vy, wz, bh, rx, ry]
                # privileged obs:
                left_touch,
                right_touch,
                feet_position_6,
                base_position_3,
                base_orientation_4,
                com_inertia_n,
                com_vel_n,
                base_lin_vel_3,
                base_ang_vel_3,
                actuator_force_n,  # TODO why was this / 4.0?
                base_height,
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _ppo_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry = actor_critic_carry
        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=critic_carry,
        )

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(model, rng),
            (next_actor_carry, next_critic_carry),
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        scan_fn = functools.partial(self._ppo_scan_fn, model=model)
        rngs = jax.random.split(rng, trajectory.done.shape[0])
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, rngs),
            jit_level=ksim.JitLevel.RL_CORE,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(
        self, model: Model, rng: PRNGKeyArray
    ) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(rng)
        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,  # was 2048, not enough memory
            batch_size=64,  # How many training samples to use per network update. More samples == more memory
            num_passes=4,
            # epochs_per_log_step=1,
            rollout_length_seconds=4.0,  # How long does each training sample last? Longer == more memory?
            # global_grad_clip=2.0,
            entropy_coef=0.004,
            # Simulation parameters.
            dt=0.004,  # The step size (in seconds) of the simulation. Lower values == more accurate physics sim, but requires more memory for the same simulation time
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.003, 0.01),  # Simulate 3-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_track_body_id=0,
            render_markers=True,
            # render_full_every_n_seconds=0,
            render_length_seconds=10,
            max_values_per_plot=50,
            # Checkpointing parameters.
            save_every_n_seconds=60,
        ),
    )
