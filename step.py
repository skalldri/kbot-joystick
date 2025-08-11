from model import Actor
from jaxtyping import Array
import jax.numpy as jnp
import distrax
import xax


def step_fn(
    actor: Actor,
    joint_angles: Array,
    joint_angular_velocities: Array,
    gyroscope: Array,
    accelerometer: Array,
    command: Array,
    carry: Array,
    **kwargs,
) -> tuple[xax.Distribution, Array]:

    obs = [
        joint_angles,  # NUM_JOINTS (20)
        joint_angular_velocities,  # NUM_JOINTS (20)
        gyroscope,  # 3 (IMU gyroscope data)
        accelerometer,  # 3 (IMU accelerometer data)
        command,  # 6: [vx, vy, yaw, body height, roll x, roll y]
    ]

    obs_n = jnp.concatenate(obs, axis=-1)

    assert len(obs_n) == 52

    action, carry = actor.forward(obs_n, carry)

    return action, carry
