from gym import ObservationWrapper
import numpy as np


def TrigonometricObservationWrapper(ObservationWrapper):

    def observation(self, observation):
        """
        With an observation of shape [theta, alpha, theta_dot, alpha_dot], this wrapper transform this
        observation to [cos(theta), sin(theta), cos(alpha), sin(alpha), theta_dot, alpha_dot]
        """
        assert len(observation) != 4, "Assumes a observation which is in shape [theta, alpha, theta_dot, alpha_dot]."

        return convert_single_state(observation)


def convert_single_state(state):
    theta, alpha, theta_dot, alpha_dot = state

    return np.array([np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha), theta_dot, alpha_dot], dtype=np.float64)


def convert_states_array(states):
    return np.concatenate((np.cos(states[:, 0:1]), np.sin(states[:, 0:1]), np.cos(states[:, 1:2]), np.sin(states[:, 1:2]), states[:, 2:3], states[:, 3:4], states[:, 4:]), axis=1)