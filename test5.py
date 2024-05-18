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
import torch.functional as F

import gym
import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# import numpy as np
from scipy import linalg
from scipy import signal
import pandas as pd
import torch
from r_actor_critic_dpo import R_Actor, R_Critic, Penalty
from algorithms.mappo.utils.util import update_linear_schedule,soft_update,hard_update
from torch.distributions import Categorical
# 几乎没变
import numpy as np
import copy
import sys
def _t2n(x):
    return x.detach().cpu().numpy()
from algorithms.mappo.utils.config import get_config
parser = get_config()
def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='pendulum', help="Which scenario to run on")
    # parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args
all_args = parse_args(sys.argv[1:], parser)
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
class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.penalty_lr = args.penalty_lr

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        # self.paramter_actionsize = args.hidden_size
        

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        # self.critic = R_Critic(args, self.share_obs_space,self.act_space, self.device)
        self.target_actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.target_critic = R_Critic(args, self.share_obs_space, self.device)
        # self.target_critic = R_Critic(args, self.share_obs_space,self.act_space, self.device)
        self.penalty = Penalty(args, self.obs_space, self.device)

        

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.penalty_optimizer = torch.optim.Adam(self.penalty.parameters(),
                                                 lr=self.penalty_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
    def get_actions(self, cent_obs, obs, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        # if target_rnn_states_actor is None:
        #     target_rnn_states_actor = np.array(rnn_states_actor)
        # print("obs",obs.shape)
        # actions, action_log_probs, rnn_states_actor = self.actor(obs,
        #                                                          rnn_states_actor,
        #                                                          masks,
        #                                                          available_actions,
        #                                                          deterministic)
        actions, action_log_probs,= self.actor(obs,
                                                                 
                                                                 available_actions,
                                                                 deterministic)

        values = self.critic(cent_obs)
        # print("action",actions.shape)
        
        return values, actions, action_log_probs,

    def get_values(self, cent_obs):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs)
        return values
    
    def get_penalty(self, cent_obs):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, rnn_states_penalty = self.penalty(cent_obs)
        # print("values",values)
        return values, rnn_states_penalty

    def evaluate_actions(self, cent_obs, obs,  action, 
                         available_actions=None, active_masks=None,deterministic=False):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     
                                                                     action,
                                                                     
                                                                     available_actions,
                                                                     active_masks)
        actions, action_log_probs = self.actor(obs,
                                                                 
                                                                 
                                                                 available_actions,
                                                                 deterministic)

        values, _ = self.critic(cent_obs)
        return values, action_log_probs, dist_entropy, actions
    

    def act(self, obs,  available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _ = self.actor(obs, available_actions, deterministic)
        return actions
    def get_probs(self, obs, 
                         available_actions=None,prob_merge=True):
        """
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_probs = self.actor.get_probs(obs,
                                            
                                            available_actions)

        # print('dist_entropy = {}'.format(dist_entropy))
        return action_probs
    
    def get_dist(self, obs,
                         available_actions=None,target=False):
        """
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if target:
            actor = self.target_actor
        else:
            actor = self.actor
        control_dist = actor.get_dist(obs,
                                                available_actions)
        

        # print('dist_entropy = {}'.format(dist_entropy))
        # print("control_dist,comm_dist",control_dist,comm_dist)
        return control_dist
    def prep_training(self):
        # sets the module into training mode. 
        # This is important for certain types of layers like Dropout and BatchNorm,
        self.actor.train()
        self.critic.train()
        self.penalty.train()
env = QubeSwingupEnv(use_simulator=True,frequency=250)
if all_args.cuda and torch.cuda.is_available():
    print("choose to use gpu...")
    device = torch.device("cuda:0")
    torch.set_num_threads(all_args.n_training_threads)
    if all_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
else:
    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)
networkclass = R_MAPPOPolicy(all_args,obs_space=env.observation_space,cent_obs_space=env.observation_space,\
                             act_space=env.action_space,device=device)
class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

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
networkclass.prep_training()
criterion = nn.MSELoss()
# for epoch in range(50):  # Number of epochs
#     for inputs, targets in train_loader:
#         networkclass.actor_optimizer.zero_grad()
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs,_ = networkclass.actor(inputs)
#         # print(outputs.requires_grad) 
#         targets = targets.to(device)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         networkclass.actor_optimizer.step()
#         # print(f'Epoch {epoch+1}, Batch Loss = {loss.item()}')  # Moved inside loop for better clarity

#         # Save model if current loss is lower
#         if loss.item() < best_val_loss:  # Ensure to use .item() to get the Python number
#             best_val_loss = loss.item()
#             save_model(networkclass.actor, 'best_modelactor.pth')
#             print("Saved Best Model",f'Loss = {loss.item()}')

networkclass.actor.load_state_dict(torch.load('best_modelactor.pth'))
with QubeSwingupEnv(use_simulator=False, frequency=250) as env:
    for episode in range(3):
        state = env.reset()
        
        # theta_dot, alpha_dot = lqr_controller.filter(state[0], state[1])
        for step in range(2048):
            state = torch.tensor(state, dtype=torch.float32)
            alpha = state[1]
            # print(alpha)
            # if np.abs(alpha) <= (5.0 * np.pi / 180.0):
            #     # print("balance")
            #     action = model_hold(state)
            #     action = action.detach().numpy()  
            #     print(action)
            #     # action = lqr_controller._action_hold(state.detach().numpy()[0],
            #     #                                      state.detach().numpy()[1],
            #     #                                      state.detach().numpy()[2],
            #     #                                      state.detach().numpy()[3],)
            #     # action =  torch.tensor(action, dtype=torch.float32)
            # else:
            action,_ = networkclass.actor(state)
            action = action.detach().cpu().numpy()       
                # action = lqr_controller.action(state)
            # action = model(state)   
            # action = action.detach().numpy()       
            state, reward, done, info = env.step(action)
            # env.render()
# env = gym.make('Pendulum-v0')
# policy = networkclass.critic
# old_policy = networkclass.target_critic
# optim = torch.optim.Adam(policy.parameters(), lr=1e-5)
# value = networkclass.actor
# value_optim = torch.optim.Adam(value.parameters(), lr=2e-5)
# gamma = 0.9
# steps = 0

# is_learn = False
# writer = SummaryWriter('ppo_logs')


# for epoch in count():
#     state = env.reset()
#     episode_reward = 0
#     rewards = []
#     states = []
#     actions = []
#     for time_steps in range(200):
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#         action = networkclass.act(state_tensor)
#         print('action : ', action)
#         next_state, reward, done, _ = env.step([action])
#         episode_reward += reward
#         # reward = (reward + 8.1) / 8.1

#         rewards.append(reward)
#         states.append(state)
#         actions.append(action)

#         state = next_state

#         if (time_steps+1) % 32 == 0 or time_steps == 199:
#             old_policy.load_state_dict(policy.state_dict())
#             with torch.no_grad():
#                 next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
#                 R = value(next_state_tensor)
#             for i in reversed(range(len(rewards))):
#                 R = gamma * R + rewards[i]
#                 rewards[i] = R
#             rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
#             for K in range(10):
#                 steps += 1
#                 state_tensor = torch.FloatTensor(states).to(device)
#                 action_tensor = torch.FloatTensor(actions).unsqueeze(1).to(device)
#                 with torch.no_grad():
#                     advantage = rewards_tensor - value(state_tensor)

#                 with torch.no_grad():
#                     old_mu, old_std = old_policy(state_tensor)
#                     old_n = Normal(old_mu, old_std)
#                 # print(value_target.shape, advantage.shape)
#                 mu, std = policy(state_tensor)
#                 # print(prob.shape)
#                 n = Normal(mu, std)
#                 log_prob = n.log_prob(action_tensor)
#                 old_log_prob = old_n.log_prob(action_tensor)
#                 ratio = torch.exp(log_prob - old_log_prob)
#                 # print(ratio.shape, log_prob.shape)
#                 L1 = ratio * advantage
#                 L2 = torch.clamp(ratio, 0.8, 1.2) * advantage
#                 # print(log_prob.shape)
#                 loss = torch.min(L1, L2)
#                 loss = - loss.mean()
#                 writer.add_scalar('action loss', loss.item(), steps)
#                 # print(loss.shape)
#                 optim.zero_grad()
#                 loss.backward()
#                 optim.step()

#                 value_loss = F.mse_loss(rewards_tensor, value(state_tensor))
#                 value_optim.zero_grad()
#                 value_loss.backward()
#                 value_optim.step()
#                 writer.add_scalar('value loss', value_loss.item(), steps)
#             rewards = []
#             states = []
#             actions = []

#     writer.add_scalar('episode reward', episode_reward, epoch)
#     if epoch % 10 == 0:
#         print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
#         torch.save(policy.state_dict(), 'ppo-policy.para')