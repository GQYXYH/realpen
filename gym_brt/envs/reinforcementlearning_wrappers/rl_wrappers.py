import numpy as np
from gym import spaces

from gym_brt.envs.qube_balance_env import QubeBalanceEnv
from gym_brt.envs.qube_swingup_env import QubeSwingupEnv

OBS_MAX = np.asarray([1, 1, 1, 1, np.inf, np.inf], dtype=np.float64)

"""
Wrapper classes for the QubeEnv of the quanser driver. The wrappers work as an OpenAi Gym interface. 

These wrappers can be used to get consistent environments with the following state representation:
[cos(theta), sin(theta), cos(alpha), sin(alpha), theta_dot, alpha_dot]. 

The reward functions provided can be found in rl_reward_functions.py. 

Wrapper always needs to be used like
    with Wrapper as wrapper:
to ensure safe closure of camera and qube!
"""

FREQUENCY = 250


class QubeBeginDownEnv(QubeSwingupEnv):
    #def __init__(self, frequency=FREQUENCY, batch_size=2048, use_simulator=False, simuilator encoder_reset_steps=int(1e8)):
    #     super().__init__(frequency, batch_size, use_simulator, encoder_reset_steps)
    #     self.observation_space = spaces.Box(-OBS_MAX, OBS_MAX, dtype=np.float32)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(-OBS_MAX, OBS_MAX, dtype=np.float32)

    def _get_state(self):
        return np.array(
            [np.cos(self._theta),
             np.sin(self._theta),
             np.cos(self._alpha),
             np.sin(self._alpha),
             self._theta_dot,
             self._alpha_dot],
            dtype=np.float64,
        )
    
class RandomStartEnv(QubeBeginDownEnv):
    def __init__(self, **kwargs):
        assert kwargs["use_simulator"], "Just possible in simulation"
        super().__init__(**kwargs)

    def reset(self):
        theta = 60 /180*np.pi*(2*np.random.rand()-1)
        alpha = 180 /180 * np.pi * (2 * np.random.rand() - 1)
        theta_vel = 2 * (2 * np.random.rand() - 1)
        alpha_vel = 2 * (2 * np.random.rand() - 1)
        return convert_state_back(np.array([theta, alpha, theta_vel, alpha_vel], dtype=np.float64))

class NoisyEnv(QubeBeginDownEnv):
    def _get_state(self):
        sigma = 0.015
        sigma_vel = 0.1

        return np.array(
            [np.cos(self._theta) + np.random.normal(0.0, sigma),
             np.sin(self._theta) + np.random.normal(0.0, sigma),
             np.cos(self._alpha) + np.random.normal(0.0, sigma),
             np.sin(self._alpha) + np.random.normal(0.0, sigma),
             self._theta_dot + np.random.normal(0.0, sigma_vel),
             self._alpha_dot + np.random.normal(0.0, sigma_vel)],
            dtype=np.float64,
        )

    def reset(self):
        super().reset()
        for _ in range(self._frequency * 3):
            self._step(4 * (np.random.rand() - 0.5))
        return self._get_state()


class QubeBeginUpEnv(QubeBalanceEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(-OBS_MAX, OBS_MAX, dtype=np.float32)

    def _get_state(self):
        return np.array(
            [np.cos(self._theta),
             np.sin(self._theta),
             np.cos(self._alpha),
             np.sin(self._alpha),
             self._theta_dot,
             self._alpha_dot],
            dtype=np.float64,
        )


def convert_state(state):
    return np.array(
        [np.arctan2(state[1], state[0]),
         np.arctan2(state[3], state[2]),
         state[4],
         state[5]],
        dtype=np.float64,
    )


def convert_state_back(state):
    return np.array(
        [np.cos(state[0]),
         np.sin(state[0]),
         np.cos(state[1]),
         np.sin(state[1]),
         state[2],
         state[3]],
        dtype=np.float64,
    )
