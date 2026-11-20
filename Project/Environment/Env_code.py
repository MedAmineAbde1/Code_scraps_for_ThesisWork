# Here as a dashboard for developping the customable reward

import inspect
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from RewardWrapper import RewardCustom ,reward_func

#env_id = "InvertedPendulum-v5"
#env_id = "HalfCheetah-v5"
#env_id = "CartPole-v1"
#env_id = "Humanoid-v5"
env_id = "HumanoidStandup-v5"
#env_id = "Walker2d-v5"

#----------------------------

# Observation function
def s2v_fn(obs):
    return np.asarray(obs, dtype=np.float32)


# Reward function
def reward_func(obs, next_obs, action, **kwargs):

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


#env = gym.make("CartPole-v1")
train_env = gym.make(env_id,)#render_mode = "human")
train_env = gym.wrappers.RecordEpisodeStatistics(train_env)
#print(env.action_space)

train_env = RewardCustom(train_env,reward_fn = reward_func)