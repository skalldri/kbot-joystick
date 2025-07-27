"""Converts a checkpoint to a deployable model."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import ksim
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

from train import HumanoidWalkingTask, Model

NUM_COMMANDS_MODEL = 6


def euler_to_quat(euler_3: Array) -> Array:
    """Converts roll, pitch, yaw angles to a quaternion (w, x, y, z).

    Args:
        euler_3: The roll, pitch, yaw angles, shape (*, 3).

    Returns:
        The quaternion with shape (*, 4).
    """
    # Extract roll, pitch, yaw from input
    roll, pitch, yaw = jnp.split(euler_3, 3, axis=-1)

    # Calculate trigonometric functions for each angle
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    # Calculate quaternion components using the conversion formula
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Combine into quaternion [w, x, y, z]
    quat = jnp.concatenate([w, x, y, z], axis=-1)

    # Normalize the quaternion
    quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)

    return quat


def rotate_quat(q1: Array, q2: Array) -> Array:
    """Rotate quaternion q1 by quaternion q2.

    Args:
        q1: quaternion to be rotated [w, x, y, z]
        q2: quaternion to rotate by [w, x, y, z]

    Returns:
        rotated quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
        ]
    )


def rotate_quat_inverse(q1: Array, q2: Array) -> Array:
    """Rotate quaternion q1 by quaternion q2.

    Args:
        q1: quaternion to be rotated [w, x, y, z]
        q2: quaternion to rotate by [w, x, y, z]

    Returns:
        rotated quaternion [w, x, y, z]
    """
    q2_conj = jnp.array([q2[0], -q2[1], -q2[2], -q2[3]])

    return rotate_quat(q1, q2_conj)


def quat_to_euler(quat_4: Array, eps: float = 1e-6) -> Array:
    """Normalizes and converts a quaternion (w, x, y, z) to roll, pitch, yaw.

    Args:
        quat_4: The quaternion to convert, shape (*, 4).
        eps: A small epsilon value to avoid division by zero.

    Returns:
        The roll, pitch, yaw angles with shape (*, 3).
    """
    quat_4 = quat_4 / (jnp.linalg.norm(quat_4, axis=-1, keepdims=True) + eps)
    w, x, y, z = jnp.split(quat_4, 4, axis=-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)

    # Handle edge cases where |sinp| >= 1
    pitch = jnp.where(
        jnp.abs(sinp) >= 1.0,
        jnp.sign(sinp) * jnp.pi / 2.0,  # Use 90 degrees if out of range
        jnp.arcsin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.concatenate([roll, pitch, yaw], axis=-1)


def get_obs(
    joint_angles: Array,
    joint_angular_velocities: Array,
    quaternion: Array,
    initial_heading: Array,
    command: Array,
    gyroscope: Array,
) -> Array:
    heading_yaw_cmd = command[2]  # yaw_heading is at index 2

    # Force the body-height parameter to

    initial_heading_quat = euler_to_quat(
        jnp.array([0.0, 0.0, initial_heading.squeeze()])
    )

    base_yaw_quat = rotate_quat_inverse(initial_heading_quat, quaternion)

    # Create quaternion from heading command
    heading_yaw_cmd_quat = euler_to_quat(jnp.array([0.0, 0.0, heading_yaw_cmd]))

    backspun_imu_quat = rotate_quat_inverse(base_yaw_quat, heading_yaw_cmd_quat)

    obs = [
        joint_angles,  # NUM_JOINTS (20)
        joint_angular_velocities,  # NUM_JOINTS (20)
        backspun_imu_quat,  # 4 (backspun IMU quaternion)
        # quaternion,
        command,  # 6: [vx, vy, yaw_heading, bh, rx, ry]
        gyroscope,  # 3 (IMU gyroscope data)
    ]

    obs_n = jnp.concatenate(obs, axis=-1)

    return obs_n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    model: Model = task.load_ckpt(ckpt_path, part="model")[0]

    # Loads the Mujoco model and gets the joint names.
    mujoco_model = task.get_mujoco_model()
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[
        1:
    ]  # Removes the root joint.

    # Constant values.
    carry_shape = (task.config.depth, task.config.hidden_size)  # (3, 128) hiddens
    # num_joints = len(joint_names)  # 20, joints outputs
    num_commands = NUM_COMMANDS_MODEL

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=num_commands,
        carry_size=carry_shape,
    )

    @jax.jit
    def init_fn() -> Array:
        """This function gets compiled into an ONNX model that can be loaded by kinfer.

        It is called by the kinfer runtime to initialize the model carry state.

        Returns:
            Array: an initialized carry with a shape compatible with the model.
        """
        return jnp.zeros(carry_shape)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        quaternion: Array,
        initial_heading: Array,
        command: Array,
        gyroscope: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        """This function gets compiled into an ONNX model that can be loaded by kinfer.
        It gets executed each time kinfer wants to execute the ML model.



        On function parameters:
        -----------------------
        kinfer will scan the function input names and match them with an appropriate data source from the ModelProvider.
        The available data sources are:

        - joint_angles
        - joint_angular_velocities
        - initial_heading
        - quaternion
        - projected_gravity
        - accelerometer
        - gyroscope
        - command
        - time
        - carry

        The names of the input variables must _exactly_ match the name of one of the data sources, or this function cannot be loaded.

        Args:
            joint_angles (Array): the joint absolute angles, one for each joint
            joint_angular_velocities (Array): the joint angular velocities, one for each joint
            quaternion (Array): the quaternion representing the robot's IMU's orientation, expected to be of shape (4,).
            initial_heading (Array): the initial heading angle of the robot, expected to be of shape (1,).
            command (Array): the command vector, expected to be of shape (6,) (for this particular model)
            gyroscope (Array): the raw gyroscope data from the IMU, expected to be of shape (3,).
            carry (Array): the carry state of the model, which is a hidden state that is passed between steps.

        Returns:
            tuple[Array, Array]: a tuple containing the model outputs (joint commands) and an updated carry state.
        """
        # heading_yaw_cmd = command[2]  # yaw_heading is at index 2
        # Because the training was done with a backspun IMU, but it was reading
        # the wrong part of the unified command, we can just pin this value to 0.0
        # since that's what the ry parameter was set to during training.
        heading_yaw_cmd = 0.0

        initial_heading_quat = euler_to_quat(
            jnp.array([0.0, 0.0, initial_heading.squeeze()])
        )

        base_yaw_quat = rotate_quat_inverse(initial_heading_quat, quaternion)

        # Create quaternion from heading command
        heading_yaw_cmd_quat = euler_to_quat(jnp.array([0.0, 0.0, heading_yaw_cmd]))

        backspun_imu_quat = rotate_quat_inverse(base_yaw_quat, heading_yaw_cmd_quat)

        # command = command.at[0].set(
        #     0.0
        # )  # Force backwards motion to see if the model is interpreting anything

        # The yaw-command thing doesn't work
        # command = command.at[2].set(0.0)

        obs = [
            joint_angles,  # NUM_JOINTS (20)
            joint_angular_velocities,  # NUM_JOINTS (20)
            # backspun_imu_quat,  # 4 (backspun IMU quaternion)
            # jnp.array([1.0, 0.0, 0.0, 0.0]),  # Force the IMU quaternion to be identity
            quaternion,
            command,  # 6: [vx, vy, yaw_heading, bh, rx, ry]
            # jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Tell the model to walk forward
            gyroscope,  # 3 (IMU gyroscope data)
        ]

        obs_n = jnp.concatenate(obs, axis=-1)

        # Try providing absolutely no input to the model and see what it does
        # obs_n = jnp.array([0.0 for _ in range(53)])

        assert len(obs_n) == 53

        dist, carry = model.actor.forward(obs_n, carry)
        return dist.mode(), carry

    init_onnx = export_fn(
        model=init_fn,
        metadata=metadata,
    )

    step_onnx = export_fn(
        model=step_fn,
        metadata=metadata,
    )

    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        metadata=metadata,
    )

    # Saves the resulting model.
    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)


if __name__ == "__main__":
    main()
