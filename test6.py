from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import gym
from gym_brt.envs import QubeSwingupEnv, QubeBalanceEnv
import torch
import torch.nn as nn

# from Qube import Qube
from time import sleep
from scipy.signal import butter, lfilter
import numpy as np
import math 
import matplotlib.pyplot as plt

from torch.distributions import Normal, Categorical
import numpy as np
from scipy import linalg
from scipy import signal
import pandas as pd
import torch.nn.functional as F
seed=6
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
# Initialize lists to store data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        theta_dot, alpha_dot =self.filter(state[0], state[1])
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
env = QubeSwingupEnv(use_simulator=True, frequency=250)
env2 = QubeSwingupEnv(use_simulator=True, frequency=250)
import wandb


run = wandb.init(entity="2017920898", \
        project="penenv",
        name="2017920898_pen_seed"+str(seed))
class LQRPPOnetworl(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LQRPPOnetworl, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Actor head: predicts action mean and log standard deviation
        self.actor_mean = nn.Linear(64, output_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, output_dim))  # Log std as a learnable parameter
        self.actor_comm = nn.Linear(64, 2)
        # Critic head: predicts value of the state
        self.critic = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Actor outputs
        mean = self.actor_mean(features)
        # print(mean.shape)
        log_std = self.actor_log_std.expand(mean.size(0), *self.actor_log_std.shape[1:])
        std = torch.exp(log_std)
        comm_logits = self.actor_comm(features)
        comm_probs = F.softmax(comm_logits, dim=-1)

        # Critic output
        value = self.critic(features)

        return mean, std, value, comm_probs

    # def act(self, x):
    #     mean, std, _ = self.forward(x)
    #     normal = Normal(mean, std)
    #     action = normal.sample()
    #     action_log_probs = normal.log_prob(action).sum(-1, keepdim=True)
    #     return action, action_log_probs
    def act(self, x, deterministic=False):
        mean, std, value, comm_probs = self.forward(x)
        if deterministic:
            action = mean
            # For deterministic actions, the log probability of the mode isn't well-defined,
            # but we can compute it assuming a very small standard deviation to avoid NaN values.
            eps = torch.full_like(std, 1e-5)  # Using a very small epsilon instead of zero to avoid NaNs
            action_log_probs = Normal(mean, eps).log_prob(mean).sum(-1, keepdim=True)
        else:
            normal = Normal(mean, std)
            action = normal.sample()
            action_log_probs = normal.log_prob(action).sum(-1, keepdim=True)
        m = Categorical(comm_probs)
        comm_action = m.sample()
        comm_log_probs = m.log_prob(comm_action)
        return action, action_log_probs, value, comm_action, comm_log_probs
    def evaluate_actions(self, x, action):
        mean, std, value,comm_probs = self.forward(x)
        normal = Normal(mean, std)
        action_log_probs = normal.log_prob(action).sum(-1, keepdim=True)
        entropy = normal.entropy().sum(-1, keepdim=True)
        

        return action_log_probs, torch.squeeze(value), entropy



state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
model = LQRPPOnetworl(input_dim=state_dim,output_dim=action_dim)
model2 = LQRPPOnetworl(input_dim=state_dim,output_dim=action_dim)
target_ppomodel = LQRPPOnetworl(input_dim=state_dim,output_dim=action_dim).to(device)
criterion = nn.MSELoss()
criterion_2 = nn.MSELoss()
optimizer_1 = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer_2 = torch.optim.Adam(model2.parameters(), lr=0.0005)
import pandas as pd

# Load the data from CSV
data = pd.read_csv('lqr_data.csv')

# Convert lists stored as strings back to Python lists
import ast
data['states'] = data['states'].apply(ast.literal_eval)
data['actions'] = data['actions'].apply(ast.literal_eval)

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# Convert lists to NumPy arrays
states = np.array(data['states'].tolist())
actions = np.array(data['actions'].tolist())

scaler_states = StandardScaler()
scaler_actions = StandardScaler()


def save_model(model, path):
    torch.save(model.state_dict(), path)
best_val_loss = float('inf')
# Prepare data for PyTorch
train_data = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(actions))
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
model.train()
model2.train()
def load_partial_state_dict(model, state_dict_path):
    # Load the state dictionary from the file
    saved_state_dict = torch.load(state_dict_path)
    
    # Filter out unnecessary keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}
    
    # Update the existing state dict
    model_state_dict.update(filtered_state_dict)
    
    # Load the updated state dict back into the model
    model.load_state_dict(model_state_dict)
load_partial_state_dict(target_ppomodel, 'best_modelPPO1.pth')
# target_ppomodel.load_state_dict(torch.load('best_modelPPO1.pth'))
def ppo_update(model, optimizer, states, actions, log_probs_old, returns, advantages, clip_param=0.2):
    # model.train()
    
    for _ in range(200):  # Number of optimization epochs
        # Recalculate outputs for updating
        action_log_probs, state_values,_ = model.evaluate_actions(states, actions)
        # print(action_log_probs,comm_log_probs)
        # print(comm_log_probs_old.shape, log_probs_old.shape)
        # print(f"Action log probs: {action_log_probs.shape}")
        # print(f"Comm log probs: {comm_log_probs.shape}")
        # print(f"Log probs old: {log_probs_old.shape}")
        # print(f"Comm log probs old: {comm_log_probs_old.shape}")
        # action_log_probs_all = torch.cat((action_log_probs,comm_log_probs.reshape(-1,1)),dim=-1)
        # log_probs_old_all= torch.cat((log_probs_old.reshape(-1,1),comm_log_probs_old.reshape(-1,1)),dim=-1)
        ratios = torch.exp(action_log_probs - log_probs_old)
        ratios = torch.mean(ratios, dim=-1)
        
        # Clipping the ratios
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - state_values).pow(2).mean()
        # print(action_log_probs.shape, log_probs_old.shape)
        kl_div = torch.mean(action_log_probs - log_probs_old)
        sqrt_kl = torch.sqrt(torch.max(kl_div + 1e-9,1e-9 * torch.ones_like(kl_div)))
        # kl_divs.append(kl_div.item())
        # sqrt_kl_divs.append(sqrt_kl.item())
        # print(kl_divs)

        # Total loss
        loss = 0.5 * critic_loss + actor_loss + 0.001*sqrt_kl+ 0.001 * kl_div
        

        # Taking gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    next_value = next_value[0]  # Ensure next_value is [1] not [1, 1]
    # print(next_value.shape, values.shape)
    values = torch.cat([values, next_value], dim=0) 
    # values = torch.cat([values, next_value.squeeze()], dim=0)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * gae * masks[step]
        returns.insert(0, gae + values[step])
    returns = torch.tensor(returns).to(device)
    advantages = returns - values[:-1]
    return returns, advantages
model = model.to(device)
model2 = model2.to(device)
blending_factor = 0
max_epreward = float('-inf')
record_reward = []
total_steps = 0 
wandb.log({"Episode Reward": 0}, step=0)
# def blend_policies(base_model, target_model, blending_factor):
#     for base_param, target_param in zip(base_model.parameters(), target_model.parameters()):
#         base_param.data.copy_((1 - blending_factor) * base_param.data + blending_factor * target_param.data)
def blend_policies(base_model, target_model, blending_factor):
    base_params = {name: param for name, param in base_model.named_parameters()}
    target_params = {name: param for name, param in target_model.named_parameters()}

    for name, base_param in base_params.items():
        if name in target_params:
            target_param = target_params[name]
            base_param.data.copy_((1 - blending_factor) * base_param.data + blending_factor * target_param.data)
for epoch in range(50):
    memory = []
    state = env.reset()
    epreward = 0
    blending_factor = min(blending_factor + 0.001, 0)  # Ensure blending factor doesn't exceed 1

    for t in range(1000):  # Collect data for 1000 steps
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action, action_log_prob, value,_,_ = model.act(state)
        action = action[0].detach().cpu().numpy()
        next_state, reward, done, _ = env.step(action)
        
        epreward += reward
        memory.append((state, action, action_log_prob, reward, 1 - done, value))
        state = next_state
        # if done:
        #     break

    print(f"Epoch {epoch+1}: Episode Reward = {epreward}")
    total_steps += t  # Increment total steps by the number of steps taken in this epoch
    wandb.log({"Episode Reward": epreward}, step=total_steps)
    record_reward.append(epreward)

    if epreward > max_epreward:
        max_epreward = epreward
        # torch.save(model.state_dict(), 'best_model.pth')
        print(f"New maximum episode reward {epreward} achieved, model saved.")

    # Prepare the data for training
    states, actions, log_probs_old, rewards, masks, values = zip(*memory)
    states = torch.stack(states).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    masks = torch.tensor(masks, dtype=torch.float32).to(device)
    values = torch.tensor(values, dtype=torch.float32).to(device)

    next_value = model(torch.tensor([state], dtype=torch.float32).to(device))[2]
    returns, advantages = compute_gae(next_value, rewards, masks, values)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)

    # Perform PPO update
    ppo_update(model, optimizer_1, states, actions, log_probs_old, returns, advantages)
    blend_policies(model, target_ppomodel, blending_factor)
run.finish()
data = pd.DataFrame({
    'EpisodeReward': record_reward
})

# Save to CSV
data.to_csv('reward.csv', index=False)
