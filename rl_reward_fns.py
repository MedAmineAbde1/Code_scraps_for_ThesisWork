import random
import math
import numpy as np 

def reward_func_humanoid_standing_v1(obs, next_obs, action, **kwargs):

    # Torso height
    height = float(next_obs[0])

    # Height progress
    height_progress = float(next_obs[0] - obs[0])

    # Torso uprightness (orientation quaternion: w, x, y, z)
    x, y = next_obs[2], next_obs[3]
    uprightness = np.clip(1.0 - 2.0 * (x**2 + y**2), -1.0, 1.0)

    # Control effort
    control_cost = float(np.mean(np.square(action)))

    return float(
        2.0 * height
        + 10.0 * height_progress
        + 5.0 * uprightness
        - 0.1 * control_cost
    )

def reward_func_Human_stand_v2(obs, next_obs, action, terminated=False, **kwargs):

    # 1. Torso height (verify observation layout)
    height = float(next_obs[0])
    prev_height = float(obs[0])

    # 2. Uprightness: quaternion order is w, x, y, z
    qx = float(next_obs[2])
    qy = float(next_obs[3])

    uprightness = np.clip(
        1.0 - 2.0 * (qx**2 + qy**2),
        -1.0, 1.0
    )

    # 3. Normalize height and uprightness to [0, 1]
    height_score = np.clip(
        (height - 0.5) / (1.3 - 0.5),
        0.0, 1.0
    )

    upright_score = (uprightness + 1.0) / 2.0

    # 4. Reward improving height
    height_progress = np.clip(
        (height - prev_height) * 10.0,
        -1.0, 1.0
    )

    # 5. Bonus for achieving BOTH height and uprightness
    standing_bonus = (
        5.0
        if height >= 1.2 and uprightness >= 0.8
        else 0.0
    )

    # 6. Penalize excessive control effort
    control_cost = float(np.mean(np.square(action)))

    # Final reward
    reward = (
        2.0 * height_score
        + 2.0 * upright_score
        + 0.5 * height_progress
        + standing_bonus
        - 0.01 * control_cost
    )

    return float(reward)


def reward_func_Human_Stand_v3(obs, next_obs, action, **kwargs):

    obs = np.asarray(obs, dtype=np.float64)
    next_obs = np.asarray(next_obs, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)

    # 1. Torso height
    prev_height = float(obs[0])
    height = float(next_obs[0])

    # 2. Torso orientation quaternion: w, x, y, z
    qw = float(next_obs[1])
    qx = float(next_obs[2])
    qy = float(next_obs[3])
    qz = float(next_obs[4])

    # Normalize quaternion to protect against numerical errors
    q = np.array([qw, qx, qy, qz])
    q_norm = np.linalg.norm(q)

    if q_norm > 1e-8:
        qw, qx, qy, qz = q / q_norm
    else:
        return 0.0

    # 3. Uprightness: +1 upright, -1 upside down
    uprightness = np.clip(
        1.0 - 2.0 * (qx**2 + qy**2),
        -1.0,
        1.0
    )

    upright_score = (uprightness + 1.0) / 2.0

    # 4. Height score
    # Reaches 1 when torso height reaches 1.6 m
    height_score = np.clip(
        (height - 0.5) / (1.6 - 0.5),
        0.0,
        1.0
    )

    # 5. Small reward for increasing torso height
    height_progress = np.clip(
        5.0 * (height - prev_height),
        -0.5,
        0.5
    )

    # 6. Reward maintaining a standing posture
    standing_bonus = (
        2.0
        if height >= 1.2 and uprightness >= 0.8
        else 0.0
    )

    # 7. Penalize excessive actions
    control_cost = 0.05 * np.sum(np.square(action))

    # 8. Penalize excessive contact forces.
    # Assumes default HumanoidStandup-v5 observation layout.
    # cfrc_ext occupies indices 270:348.
    if len(next_obs) >= 348:
        contact_forces = next_obs[270:348]

        impact_cost = 5e-7 * np.clip(
            np.sum(np.square(contact_forces)),
            0.0,
            10.0
        )
    else:
        impact_cost = 0.0

    # 9. Final reward
    reward = (
        1.5 * height_score
        + 2.0 * upright_score
        + 0.5 * height_progress
        + standing_bonus
        - control_cost
        - impact_cost
    )

    return float(reward)