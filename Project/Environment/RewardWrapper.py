# Injects a customizable reward function into Gymnasium.
# Receives transitions and calculates the reward returned
# to PPO, independently of the original environment reward.



# Here as a dashboard for developping the customable reward

import inspect
import gymnasium as gym
from gymnasium import spaces
import numpy as np

#env_id = "InvertedPendulum-v5"
#env_id = "HalfCheetah-v5"
#env_id = "CartPole-v1"
#env_id = "Humanoid-v5"
env_id = "HumanoidStandup-v5"
#env_id = "Walker2d-v5"

### custom reward

class RewardCustom(gym.Wrapper):
    """
    reward_fn can take any subset of these argument names (or **kwargs for all):
    obs, action, next_obs, reward, terminated, truncated, info, env
    """
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn
        params = inspect.signature(reward_fn).parameters
        self._takes_all = any(p.kind == p.VAR_KEYWORD for p in params.values())
        self._params = set(params)
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs = obs
        return obs, info

    def step(self, action):
        u = self.env.unwrapped
        state = getattr(u, "state",None) # None for envs without .state

        next_obs, reward, terminated, truncated, info = self.env.step(action)

        next_state = getattr(u, "state",None)

        ctx = dict(
            state=state,
            action=action,
            next_state=next_state,
            obs=self._obs,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            env=u,
        )

        kwargs = (
            ctx if self._takes_all
            else {k: v for k, v in ctx.items() if k in self._params}
        )

        info["original_reward"] = reward

        new_reward = float(self.reward_fn(**kwargs))

        self._obs = next_obs

        return next_obs, new_reward, terminated, truncated, info





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