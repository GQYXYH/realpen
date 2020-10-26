"""
Reinforcement learning reward functions for the different tasks Balance, Swing Up and General. Simple rewards and
energy based rewards can be found in the functions and need to be selected or modified.
"""
import numpy as np
import typing as tp
import math


def angle_normalize(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


def swing_up_reward(theta: float, alpha: float, target_angle: float = 0.0):
    reward = 1 - (
            (0.8 * np.abs(alpha) + 0.2 * np.abs(target_angle - theta))
            / np.pi
    )

    if abs(theta) > (80 * np.pi / 180):
        reward -= 1

    # if np.abs(alpha) < 5 / 180 * np.pi: reward *= 2
    # if np.abs(alpha) < 3 / 180 * np.pi: reward *= 2
    return max(reward, 0) ** 2  # Clip for the follow env case

    # alpha_reward = 1 - alpha/np.pi**2
    # alpha_reward += max(0, 0.3-(alpha/np.pi/0.06)**2)
    # theta_reward = 1-theta/np.pi**2
    # return 4*alpha_reward + theta_reward

    # theta_reward = 1-(theta/(0.33*np.pi))**4
    # alpha_reward = 4*np.exp(-3*np.abs(alpha/np.pi))
    #
    # return theta_reward + alpha_reward


def extended_swing_up_reward(state, action: float, target_angle: float = 0.0):
    theta, alpha, theta_dot, alpha_dot = state
    cost = angle_normalize(alpha) ** 2
    cost += 5e-3 * alpha_dot ** 2
    cost += 1e-1 * np.abs(theta - target_angle) ** 2
    cost += 2e-2 * theta_dot ** 2
    cost += 3e-3 * action ** 2
    return -cost


def cosine_swing_up_reward(theta: float, alpha: float, weight: float = 0.8):
    return weight * np.cosine(alpha) + (1.0 - weight) * np.cosine(theta)


def exp_swing_up_reward(state, action: float, dt: float):
    """
    Adapted from
    https://git.ias.informatik.tu-darmstadt.de/quanser/clients/-/blob/master/quanser_robots/qube/base.py

    Args:
        state:      Current state of the Qube-Servo 2 in shape (theta, alpha, theta_dot, alpha_dot)
        action:     Last taken action
        dt:         Time difference between two control steps (== 1.0 / frequency)

    Returns:        Calculated reward
    """
    theta, alpha, theta_dot, alpha_dot = state
    cost = alpha ** 2 + 5e-3 * alpha_dot ** 2 + 1e-1 * theta ** 2 + 2e-2 * theta_dot ** 2 + 3e-3 * action ** 2
    if abs(theta) > (90.0 * np.pi / 180.0):
        return -1
    else:
        return math.exp(-cost) #* dt


def balance_reward(theta, alpha, target_angle):
    reward = 1 - (
            (0.8 * np.abs(alpha) + 0.2 * np.abs(target_angle - theta))
            / np.pi
    )
    return max(reward, 0)  # Clip for the follow env case

# class GeneralReward(object):
#     def __init__(self):
#         self.target_space = spaces.Box(-ACT_MAX, ACT_MAX, dtype=np.float32)
#
#     def __call__(self, state, action):
#         theta_x = state[0]
#         theta_y = state[1]
#         alpha_x = state[2]
#         alpha_y = state[3]
#         theta_velocity = state[4]
#         alpha_velocity = state[5]
#         theta_acceleration = state[6]
#         alpha_acceleration = state[7]
#
#         params = np.arctan2(theta_y, theta_x)  # arm
#         alpha = np.arctan2(alpha_y, alpha_x)  # pole
#
#         # # By Hand Reard
#         # cost = 5 * normalize_angle(params) ** 10 + \
#         #         normalize_angle(alpha) ** 2 + \
#         #         0.01 * action ** 2 + \
#         #         0.0001 * alpha_velocity ** 2 * max(0,(-abs(normalize_angle(alpha))+1.57)/1.57) ** 2
#         #         # penalize velocity of alpha if above 90 degrees
#         # cost = float(cost)
#         # reward = -cost
#
#         # Energy Based Reward
#         mr = 0.095
#         # Total length (m)
#         r = 0.085
#         # Moment of inertia about pivot (kg-m^2)
#         Jr = mr * r ** 2 / 3  # Jr = Mr*r^2/12
#         mp = 0.024  # Mass (kg)
#         Lp = 0.129  # Total length (m)
#         Jp = mp * Lp ** 2 / 3  # Moment of inertia about pivot (kg-m^2)
#         e_pot = mp * 9.81 * Lp * (1 + np.cos(alpha))
#         e_kin = 0.5 * Jp * state[5] ** 2 + 0.5 * Jr * state[4]
#         e = e_pot + e_kin
#         reward = e_pot - e_kin * max(0, (-abs(normalize_angle(alpha)) + 1) / 1) ** 2 \
#                  - 0.002 * normalize_angle(params) ** 2
#         reward = float(reward)
#
#         return reward
#
#
# class SwingUpReward(object):
#     def __init__(self):
#         self.target_space = spaces.Box(-ACT_MAX, ACT_MAX, dtype=np.float32)
#
#     def __call__(self, state, action):
#         theta_x = state[0]
#         theta_y = state[1]
#         alpha_x = state[2]
#         alpha_y = state[3]
#         theta_velocity = state[4]
#         alpha_velocity = state[5]
#         theta_acceleration = state[6]
#         alpha_acceleration = state[7]
#
#         params = np.arctan2(theta_y, theta_x)  # arm
#         alpha = np.arctan2(alpha_y, alpha_x)  # pole
#
#         cost = 5 * normalize_angle(params) ** 10 + \
#                normalize_angle(alpha) ** 2
#
#         reward = -cost
#         return reward
#
#
# class BalanceReward(object):
#     def __init__(self):
#         self.target_space = spaces.Box(-ACT_MAX, ACT_MAX, dtype=np.float32)
#
#     def __call__(self, state, action):
#         theta_x = state[0]
#         theta_y = state[1]
#         alpha_x = state[2]
#         alpha_y = state[3]
#         theta_velocity = state[4]
#         alpha_velocity = state[5]
#         theta_acceleration = state[6]
#         alpha_acceleration = state[7]
#
#         params = np.arctan2(theta_y, theta_x)  # arm
#         alpha = np.arctan2(alpha_y, alpha_x)  # pole
#
#         # simple cost function
#         cost = 50 * normalize_angle(alpha) ** 2 - 5 + 0.5*(100 * normalize_angle(params) ** 4 - 1) #- abs(action[0])*0.1
#
#
#         # # Energy Based Reward
#         # mr = 0.095
#         # r = 0.085
#         # Jr = mr * r ** 2 / 3
#         # mp = 0.024
#         # Lp = 0.129
#         # Jp = mp * Lp ** 2 / 3
#         # e_pot = mp * 9.81 * Lp * (1 + np.cos(alpha))
#         # e_kin = 0.5 * Jp * state[5] ** 2 + 0.5 * Jr * state[4]
#         # e = e_pot + e_kin
#         # cost = e_kin - 5*e_pot
#         # cost = float(cost)
#
#         # # LQR cost function
#         # Q = np.eye(4)
#         # Q[0, 0] = 12
#         # Q[1, 1] = 5
#         # Q[2, 2] = 1
#         # R = np.array([[1]]) * 1
#         # x = np.array([params, alpha, alpha_velocity, theta_velocity])
#         # u = np.array([action])
#         # cost = x.dot(Q).dot(x) + u.dot(R).dot(u)
#         # cost = float(cost)
#
#         reward = -cost
#         return reward
