import numpy as np
from gym import spaces

from gym_brt.envs.qube_base_env import normalize_angle, ACT_MAX

"""
Reinforcement learning reward functions for the different tasks Balance, Swing Up and General. Simple rewards and 
energy based rewards can be found in the functions and need to be selected or modified.
"""


def swing_up_reward(theta, alpha, target_angle):
    reward = 1 - (
            (0.8 * np.abs(alpha) + 0.2 * np.abs(target_angle - theta))
            / np.pi
    )
    return max(reward, 0)  # Clip for the follow env case


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
