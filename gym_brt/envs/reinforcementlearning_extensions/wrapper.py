"""
Wrapper for OpenAI Gym Reinforcement Learning environments

@Author: Moritz Schneider
"""
import typing as tp

import cv2
import numpy as np
from gym import ObservationWrapper, Wrapper, spaces, Env

from gym_brt.control import calibrate, PIDCtrl
from gym_brt.data.config.configuration import FREQUENCY

Array = tp.Union[tp.List, np.ndarray]


class ImageObservationWrapper(ObservationWrapper):
    """
    Wrapper to get an image from the environment and not a state.
    Use env.render(rgb_array) as observation rather than the observation environment provides
    """

    def __init__(self, env, out_shape=None):
        super(ImageObservationWrapper, self).__init__(env)
        dummy_obs = env.render("rgb_array")
        # Update observation space
        self.out_shape = out_shape

        obs_shape = out_shape if out_shape is not None else dummy_obs.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=dummy_obs.dtype)

    def observation(self, observation: np.ndarray):
        img = self.env.render("rgb_array")
        if self.out_shape is not None:
            img = cv2.resize(img, (self.out_shape[0], self.out_shape[1]), interpolation=cv2.INTER_AREA)
        # TODO: Resulting state as information?
        return img


class TrigonometricObservationWrapper(ObservationWrapper):

    def observation(self, observation: Array):
        """
        With an observation of shape [theta, alpha, theta_dot, alpha_dot], this wrapper transform this
        observation to [cos(theta), sin(theta), cos(alpha), sin(alpha), theta_dot, alpha_dot]
        """
        assert len(observation) != 4, "Assumes a observation which is in shape [theta, alpha, theta_dot, alpha_dot]."

        return convert_single_state(observation)


def convert_single_state(state: Array):
    theta, alpha, theta_dot, alpha_dot = state

    return np.array([np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha), theta_dot, alpha_dot], dtype=np.float64)


def convert_states_array(states: Array):
    return np.concatenate((np.cos(states[:, 0:1]), np.sin(states[:, 0:1]), np.cos(states[:, 1:2]), np.sin(states[:, 1:2]), states[:, 2:3], states[:, 3:4], states[:, 4:]), axis=1)


class CalibrationWrapper(Wrapper):
    """Wrapper to calibrate the rotary arm of the Qube to a specific angle."""

    def __init__(self, env: Env, desired_theta: float = 0.0, frequency: int = None, u_max: float = 1.0,
                 noise: bool = False, unit='deg', save_limits=True, limit_reset_threshold=None):
        super(CalibrationWrapper, self).__init__(env)
        self.frequency = FREQUENCY if frequency is None else frequency
        self.u_max = u_max
        self.desired_theta = desired_theta
        self.noise = noise
        self.save_limits = save_limits
        self.limits = None
        self.counter = 0

        self.limit_reset_threshold = np.inf if limit_reset_threshold is None else limit_reset_threshold

        if unit == 'deg':
            self.noise_scale = 45.
        elif unit == 'rad':
            self.noise_scale = np.pi/4
        else:
            self.noise_scale = 0.

    def reset(self, explore_limit=True, **kwargs):
        # First reset the env to be sure the environment it is ready for calibration
        self.env.reset(**kwargs)

        # Inject noise
        theta = self.desired_theta + np.random.normal(scale=self.noise_scale) if self.noise else self.desired_theta

        # Calibrate
        self.limits = calibrate(theta, self.frequency, self.u_max, limits=self.limits)
        self.counter += 1

        # Check if we have to reset the limits for calibration
        if self.counter >= self.limit_reset_threshold:
            self.limits = None
            self.counter = 0

        # Second reset to get the state and initialize correctly
        return self.env.reset(**kwargs)

