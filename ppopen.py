import sys
import os
sys.path.append(os.getcwd())

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

from vec_env import create_vecenv
from utils import all_seed, plot_rewards
from gym_brt.envs import QubeSwingupEnv, QubeBalanceEnv
import numpy as np
from scipy import linalg
from scipy import signal
import pandas as pd
from time import sleep
from scipy.signal import butter, lfilter
import numpy as np
import math 
import matplotlib.pyplot as plt
import gym
from gym_brt.envs import QubeSwingupEnv
from gym_brt.control import QubeFlipUpControl

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
        if np.abs(alpha) <= (20.0 * np.pi / 180.0):
            # print("balance")
            action = self._action_hold(theta, alpha, theta_dot, alpha_dot)
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

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, std=0.0):
        super(Model, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # self.log_std = nn.Parameter(torch.ones(1, output_dim) * std)
        self.log_std = nn.Parameter(torch.ones(output_dim) * std)
        # self.apply(init_linear_weights)
        
    def forward(self, x):
        # print("x state",x.device)
        value = self.critic(x)
        mu = self.actor(x)
        mu = torch.tanh(mu)
        # print("mu.shape",mu.shape)
        std = self.log_std.exp().expand_as(mu)
        # print("std",std.shape)
        return mu, std, value
    

    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        try:
            self.load_state_dict(torch.load(path))
            print("Model loaded successfully from", path)
            return True
        except FileNotFoundError:
            print("Model file not found:", path)
            return False
        except Exception as e:
            print("Error loading the model:", e)
            return False

class Policy:
    def __init__(self, cfg):
        self.model = Model(cfg.state_dim, cfg.action_dim, cfg.hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.device = torch.device(cfg.device)
        self.action_scale = (cfg.action_space.high[0] - cfg.action_space.low[0]) / 2
        self.action_bias = (cfg.action_space.high[0] + cfg.action_space.low[0]) / 2
        self.n_epochs = cfg.n_epochs
        self.mini_batch_size = cfg.mini_batch_size
        self.clip_param = cfg.clip_param
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.model.to(self.device)
        
    def get_action(self, state):
        # print(self.device)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # print("Original state device:", state.device)
        mu, std, value = self.model(state)
        mu = mu * self.action_scale + self.action_bias
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob, value
    
    @staticmethod
    def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
        
        states = states.reshape(100,4)
        actions = actions.reshape(100,1)
        log_probs= log_probs.reshape(100,1)
        returns = returns.reshape(100,1)
        advantage = advantage.reshape(100,1)
        batch_size = states.size(0)

        # print(f"Shapes - states: {states.shape}, actions: {actions.shape}, log_probs: {log_probs.shape}, returns: {returns.shape}, advantages: {advantage.shape}")
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
    def compute_gae(self, next_value, rewards, masks, values):
        # print(next_value.shape)
        # Ensure all inputs are at least two-dimensional (batch, 1)
        rewards = torch.cat(rewards, dim=0)
        masks = torch.cat(masks, dim=0)
        values = torch.cat(values, dim=0)

        next_value = next_value.unsqueeze(-1) if next_value.dim() == 1 else next_value
        rewards = rewards.unsqueeze(-1) if rewards.dim() == 1 else rewards
        masks = masks.unsqueeze(-1) if masks.dim() == 1 else masks
        values = values.unsqueeze(-1) if values.dim() == 1 else values
        # print(values.shape)
        # print(next_value.shape)

        # Append next_value to values for simplified delta calculation
        if next_value.dim() == 1:
            next_value = next_value.unsqueeze(0)  # Add the batch dimension if missing
        elif next_value.dim() == 0:
            next_value = next_value.unsqueeze(0).unsqueeze(0)  # Add both dimensions if it's a scalar

        # Concatenate next_value to values
        values = torch.cat([values, next_value], dim=0)

        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])

        returns = torch.cat(returns, dim=0) if returns else torch.tensor([], dtype=torch.float, device=cfg.device)
        return returns

    def save_model(self, filepath):
        self.model.save(filepath)
    def load_model(self, path):
        if self.model.load(path):
            print("Model loaded successfully.")
            return self
        else:
            print("Failed to load model.")
            return None


    def learn(self, **kwargs):
        states, actions, log_probs, returns, advantages = kwargs['states'], kwargs['actions'], kwargs['log_probs'], kwargs['returns'], kwargs['advantages']
        for _ in range(self.n_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(self.mini_batch_size, states, actions, log_probs, returns, advantages):
                mu, std, value = self.model(state)
                mu = mu * self.action_scale + self.action_bias
                dist = Normal(mu, std)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
        

def evaluate_policy(env, policy, vis=False):
    state = env.reset()
    if vis: env.render()
    terminated = False
    total_reward = 0
    while True:
        action, _, _ = policy.get_action(np.array(state).reshape(1, -1))
        next_state, reward,  truncated , _ = env.step(action[0])
        state = next_state
        # if vis: env.render()
        total_reward += reward
        if truncated:
            break
    
    return total_reward

class Config:
    def __init__(self):
        self.env_name = "Pendulum-v1"
        self.n_envs = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_frames = 1000000
        self.seed = 1
        self.hidden_dim = 256
        self.lr = 3e-4
        self.n_steps = 100
        self.mini_batch_size = 50
        self.n_epochs = 4
        self.threshold_reward = 10
        self.clip_param = 0.2
        self.gamma = 0.95
        self.tau = 0.95
        self.best_model_path = "best_ppomodel.pth"
        

cfg = Config()
all_seed(cfg.seed)
env = QubeBalanceEnv(use_simulator=True, frequency=250)
# env = create_vecenv(env , n_envs = cfg.n_envs)
test_env =  QubeBalanceEnv(use_simulator=True, frequency=250)
# env = create_vecenv(cfg.env_name, n_envs = cfg.n_envs)
# test_env = gym.make(cfg.env_name)
setattr(cfg, "action_space", test_env.action_space)
setattr(cfg, "state_dim", env.observation_space.shape[0])
setattr(cfg, "action_dim", env.action_space.shape[0])
policy = Policy(cfg)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
# model = LQRPPOnetworl(input_dim=state_dim,output_dim=action_dim)
# model_hold = LQRPPholdOnetworl(input_dim=state_dim,output_dim=action_dim)
criterion = nn.MSELoss()
criterion_2 = nn.MSELoss()
# optimizer_1 = torch.optim.Adam(model.parameters(), lr=0.0005)
# optimizer_2 = torch.optim.Adam(model_hold.parameters(), lr=0.0005)
import pandas as pd

# Load the data from CSV
data = pd.read_csv('lqr_data.csv')
data_2 = pd.read_csv('lqr_holddata.csv')

# Convert lists stored as strings back to Python lists
import ast
data['states'] = data['states'].apply(ast.literal_eval)
data['actions'] = data['actions'].apply(ast.literal_eval)
data_2['states'] = data_2['states'].apply(ast.literal_eval)
data_2['actions'] = data_2['actions'].apply(ast.literal_eval)
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# Convert lists to NumPy arrays
states = np.array(data['states'].tolist())
actions = np.array(data['actions'].tolist())
states_2 = np.array(data_2['states'].tolist())
actions_2 = np.array(data_2['actions'].tolist())
scaler_states = StandardScaler()
scaler_actions = StandardScaler()
scaler_states.fit(states_2)
scaler_actions.fit(actions_2)
states_2 = scaler_states.transform(states_2)
actions_2 = scaler_actions.transform(actions_2)

def save_model(model, path):
    torch.save(model.state_dict(), path)
best_val_loss = float('inf')
# Prepare data for PyTorch
train_data = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(actions))
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)

frame_idx  = 0
test_rewards = []
test_frames = []
state = env.reset()
early_stop = False
best_reward = float('-inf')
while frame_idx < cfg.max_frames and not early_stop:

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0

    for _ in range(cfg.n_steps):
        action, log_prob, value = policy.get_action(state)
        # next_state, reward, terminated, truncated, _ = env.step(action)
        next_state, reward, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float, device=cfg.device))
        masks.append(torch.tensor([1.0 - truncated], dtype=torch.float, device=cfg.device))

        
        states.append(torch.tensor(state, dtype=torch.float, device=cfg.device))
        actions.append(torch.tensor(action, dtype=torch.float, device=cfg.device))
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 500 == 0:
            test_reward = np.mean([evaluate_policy(test_env, policy) for _ in range(10)])
            test_rewards.append(test_reward)
            test_frames.append(frame_idx)
            print(f"Frame {frame_idx}. Mean reward: {test_reward}")
            # plot_rewards(test_frames, test_rewards, device = cfg.device)
            if test_reward > cfg.threshold_reward: early_stop = True
            if test_reward > best_reward:
                best_reward = test_reward
                policy.save_model(cfg.best_model_path)  # Save the best model
                print("Saved new best model with reward:", test_reward)
        
    next_state = torch.FloatTensor(next_state).to(cfg.device)
    _, _, next_value = policy.model(next_state)
    # print(len(rewards))
    returns = policy.compute_gae(next_value, rewards, masks, values)
    # print(returns.shape)

    returns = returns.detach()

    # Ensure that log_probs, values, states, and actions are tensors and concatenate if they are lists
    log_probs = torch.cat(log_probs).detach() if isinstance(log_probs, list) else log_probs.detach()
    values = torch.cat(values).detach() if isinstance(values, list) else values.detach()
    states = torch.cat(states) if isinstance(states, list) else states
    actions = torch.cat(actions) if isinstance(actions, list) else actions
    advantage = returns - values
    # print(f"Shapes - states: {states.shape}, actions: {actions.shape}, log_probs: {log_probs.shape}, returns: {returns.shape}, advantages: {advantage.shape}")

    policy.learn(states=states, actions=actions, log_probs=log_probs, returns=returns, advantages=advantage)
env.close()
test_env.close()
load_model = policy.load_model(cfg.best_model_path)
for epoch in range(50):  # Number of epochs
    for inputs, targets in train_loader:
        optimizer_1.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_1.step()
        # print(f'Epoch {epoch+1}, Batch Loss = {loss.item()}')  # Moved inside loop for better clarity

        # Save model if current loss is lower
        if loss.item() < best_val_loss:  # Ensure to use .item() to get the Python number
            best_val_loss = loss.item()
            save_model(model, 'best_modelppo2.pth')
            print("Saved Best Model",f'Loss = {loss.item()}')
# print(load_model)
# lqr_controller = LQR(1,250)
if load_model is None:
    print("Model failed to load.")
else:
    print("Model loaded successfully.")

with QubeSwingupEnv(use_simulator=False, frequency=250) as env:
    for step in range(1024):
        state = torch.tensor(state, dtype=torch.float32)
        alpha = state[1]
        # print(alpha)
        if np.abs(alpha) <= (20.0 * np.pi / 180.0):
            # print("balance")
            action = load_model.get_action(state)[0]
            # print(action)
            # action = action.detach().numpy()       

            # print(action)
            # action = lqr_controller._action_hold(state.detach().numpy()[0],
            #                                      state.detach().numpy()[1],
            #                                      state.detach().numpy()[2],
            #                                      state.detach().numpy()[3],)
            # action =  torch.tensor(action, dtype=torch.float32)
        else:
            action = lqr_controller.action(state)
        # action = model(state)   
        state, reward, done, info = env.step(action)
        env.render()
