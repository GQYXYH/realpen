from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import gym
from gym_brt.envs import QubeSwingupEnv,QubeBalanceEnv
from gym_brt.control import QubeFlipUpControl
# from Qube import Qube
from time import sleep
from scipy.signal import butter, lfilter
import numpy as np
import math 
import matplotlib.pyplot as plt
import gym
from gym_brt.envs import QubeSwingupEnv
from gym_brt.control import QubeFlipUpControl


import numpy as np
from scipy import linalg
from scipy import signal
import pandas as pd

# Initialize lists to store data
states = []
actions = []
class LQR:
    def __init__(self, k, fs):
        self.k = k
        self.fs = fs
        self.sample_freq = fs
        self.period = 1/self.fs

        self.theta_values = []
        self.alpha_values = []
        self.voltage_values = []

        self.previous_theta = 0
        self.previous_alpha = 0

        self.theta_dot_values = []
        self.alpha_dot_values = []
        self.window = 100

        self.theta_dot_filtered_values = [0] * (self.window-1)
        self.alpha_dot_filtered_values = [0] * (self.window-1)

        self.mp = 0.024
        self.g = 9.81
        self.Lp = 0.129
        self.l =  self.Lp/2
        self.Jp = self.mp * (self.Lp ** 2) / 3 
        self.Jp_cm = self.mp*self.Lp**2/12
        self.Er = self.mp * self.g * self.l * 2

        print("LQR Controller Initialized...\n")

        self.kp_theta, self.kp_alpha, self.kd_theta, self.kd_alpha = self._get_optimized_gains()

    def _get_optimized_gains(self):
        kp_theta, kp_alpha, kd_theta, kd_alpha = self._calculate_lqr(self.sample_freq / 1.2)
        kp_theta, _, kd_theta, _ = self._calculate_lqr(self.sample_freq * 1.1)

        # Parameters tuned by hand
        if self.sample_freq < 120:
            kp_theta = -2.3  # -2.1949203339944114
            kd_theta = -1.13639961510041033
        elif self.sample_freq < 80:
            print("Critical sample frequency! LQR not tuned for frequencies below 80 Hz.")
        return kp_theta, kp_alpha, kd_theta, kd_alpha
    def _dlqr(self, A, B, Q, R):
        """
        Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # first, solve the ricatti equation
        P = np.array(linalg.solve_discrete_are(A, B, Q, R))
        # compute the LQR gain
        K = np.array((linalg.inv(B.T.dot(P).dot(B) + R)).dot(B.T.dot(P).dot(A)))
        return K

    def _calculate_lqr(self, freq=None):
        if freq is None:
            freq = self.sample_freq
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 149.2751, -0.0104, 0], [0, 261.6091, -0.0103, 0]])
        B = np.array([[0], [0], [49.7275], [49.1493]])
        C = np.array([[1, 0, 0, 0]])
        D = np.array([[0]])
        (Ad, Bd, Cd, Dd, dt) = signal.cont2discrete((A, B, C, D), 1 / freq, method='zoh')

        Q = np.eye(4)
        Q[0, 0] = 12
        Q[1, 1] = 5
        Q[2, 2] = 1
        R = np.array([[1]]) * 1

        K = self._dlqr(Ad, Bd, Q, R)
        kp_theta = K[0, 0]
        kp_alpha = K[0, 1]
        kd_theta = K[0, 2]
        kd_alpha = K[0, 3]
        return kp_theta, kp_alpha, kd_theta, kd_alpha
    def derivative(self, current, previous):
        return (current - previous) / self.period
    

    def _butter_lowpass(self, cutoff, fs, order=3):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(self, data, cutoff, order=3):
        b, a = self._butter_lowpass(cutoff, self.fs, order=order)
        y = lfilter(b, a, data)
        return y
    def _action_hold(self, theta, alpha, theta_dot, alpha_dot):
        # multiply by proportional and derivative gains
        action = \
            theta * self.kp_theta + \
            alpha * self.kp_alpha + \
            theta_dot * self.kd_theta + \
            alpha_dot * self.kd_alpha
        return action

    def action(self, state):

        # theta, alpha, theta_dot, alpha_dot = state
        theta_dot, alpha_dot = lqr_controller.filter(state[0], state[1])
        alpha = state[1]
        theta = state[0]
        # If pendulum is within 20 degrees of upright, enable balance control
        if np.abs(alpha) <= (25.0 * np.pi / 180.0):
            # print("balance")
            action = self._action_hold(theta, alpha, theta_dot, alpha_dot)
            states.append(state)
            actions.append([action])
        else:
            action = self.swingup_update(state[1], alpha_dot)

        voltages = np.array([action], dtype=np.float64)

        # set the saturation limit to +/- the Qube saturation voltage
        np.clip(voltages, -15, 15, out=voltages)
        # assert voltages.shape == self.action_shape
        return voltages
    
    def filter(self,theta,alpha):
        self.alpha_values.append(alpha)
        self.theta_values.append(theta)

        alpha_dot = self.derivative(alpha,self.previous_alpha)
        theta_dot = self.derivative(theta,self.previous_theta)

        self.alpha_dot_values.append(alpha_dot)
        self.theta_dot_values.append(theta_dot)

        if len(self.alpha_dot_values) >= self.window:

            alpha_dot = float(self.butter_lowpass_filter(self.alpha_dot_values[-self.window:],50)[-1])
            theta_dot = float(self.butter_lowpass_filter(self.theta_dot_values[-self.window:],50)[-1])
            self.theta_dot_filtered_values.append(theta_dot)
            self.alpha_dot_filtered_values.append(alpha_dot)
        

        self.previous_alpha = alpha
        self.previous_theta = theta

        return theta_dot, alpha_dot,


    def balance_update(self,x):
        u = -1 * np.dot(self.k, x)
        return u
    
    def swingup_update(self,alpha, alpha_dot):
        E = self.calculate_kinetic_energy(alpha_dot) + self.calculate_potential_energy(alpha)
        energy_error = E - 3*self.Er
        control_gain = 10  
        return control_gain * energy_error * self.signum(alpha_dot * math.cos(alpha))

    def calculate_kinetic_energy(self,alpha_dot):
        return 0.5 * self.Jp_cm * alpha_dot ** 2
    
    def calculate_potential_energy(self,alpha):
        return self.mp * self.g * self.l * (1 - math.cos(alpha))
    
    @staticmethod
    def signum(value):
        return int(value > 0) - int(value < 0)
    
    def saturation(self, threshold, input):
        u = input
        if input >= threshold:
            u = threshold
        elif input <= -threshold:
            u = -threshold
        
        return u
    
    def plot(self):
        
        time_vector = np.linspace(0, len(self.theta_values) * self.period, len(self.theta_values))

        print("Plotting...")
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(time_vector,self.theta_values, label='Theta')
        plt.plot(time_vector,self.alpha_values, label='Alpha')

        plt.title('Theta and Alpha')
        plt.ylabel('Radians')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)

        plt.subplot(212)
        plt.plot(time_vector,self.voltage_values, label='Voltage')
        plt.title('Voltage')
        plt.ylabel('V')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True)
        
        plt.show()
lqr_controller = LQR(1,250)
with QubeSwingupEnv(use_simulator=False, frequency=250) as env:
    # controller = QubeFlipUpControl(sample_freq=250, env=env)
    for episode in range(3):
        state = env.reset()
        
        # theta_dot, alpha_dot = lqr_controller.filter(state[0], state[1])
        for step in range(2048):
            action = lqr_controller.action(state)
            action = lqr_controller.saturation(20, action)            
            state, reward, done, info = env.step(action)
            # states.append(state)
            # actions.append(action)
            
# with QubeBalanceEnv(use_simulator=False, frequency=250) as env:
#     # controller = QubeFlipUpControl(sample_freq=250, env=env)
#     for episode in range(3):
#         state = env.reset()
#         # theta_dot, alpha_dot = lqr_controller.filter(state[0], state[1])
#         for step in range(2048):
#             action = lqr_controller.action(state)
#             action = lqr_controller.saturation(20, action)  
#             # print(action)          
#             state, reward, done, info = env.step(action)
            # states.append(state)
            # actions.append(action)
            
            # env.render()
            # theta_dot, alpha_dot = lqr_controller.filter(state[0], state[1])
data = pd.DataFrame({
    'states': [list(s) for s in states],  # Ensuring each state is a list (if it's not already)
    'actions': [list(a) for a in actions]  # Ensuring each action is a list (if it's not already)
})

# Save to CSV
data.to_csv('lqr_holddata.csv', index=False)
