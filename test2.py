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
        self.log_std = nn.Parameter(torch.ones(1, output_dim) * std)
        
        # self.apply(init_linear_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        mu = torch.tanh(mu)
        std = self.log_std.exp().expand_as(mu)
        return mu, std, value
    

    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path))

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
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        mu, std, value = self.model(state)
        mu = mu * self.action_scale + self.action_bias
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob, value
    
    @staticmethod
    def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
    def compute_gae(self, next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    def save_model(self, filepath):
        self.model.save(filepath)
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
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
    state,_ = env.reset()
    if vis: env.render()
    terminated = False
    total_reward = 0
    while True:
        action, _, _ = policy.get_action(np.array(state).reshape(1, -1))
        next_state, reward, terminated, truncated , _ = env.step(action[0])
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
        self.lr = 1e-3
        self.n_steps = 100
        self.mini_batch_size = 50
        self.n_epochs = 4
        self.threshold_reward = -200
        self.clip_param = 0.2
        self.gamma = 0.95
        self.tau = 0.95
        self.best_model_path = "best_model.pth"
        

cfg = Config()
all_seed(cfg.seed)
env = create_vecenv(cfg.env_name, n_envs = cfg.n_envs)
test_env = gym.make(cfg.env_name)
setattr(cfg, "action_space", test_env.action_space)
setattr(cfg, "state_dim", env.observation_space.shape[0])
setattr(cfg, "action_dim", env.action_space.shape[0])
policy = Policy(cfg)

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
        next_state, reward, terminated, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
        masks.append(torch.FloatTensor(1 - truncated).unsqueeze(1).to(cfg.device))
        
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
    returns = policy.compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states) 
    actions   = torch.cat(actions)
    advantage = returns - values
    # print(f"Shapes - states: {states.shape}, actions: {actions.shape}, log_probs: {log_probs.shape}, returns: {returns.shape}, advantages: {advantage.shape}")
    policy.learn(states=states, actions=actions, log_probs=log_probs, returns=returns, advantages=advantage)
env.close()
test_env.close()
# policy.load(cfg.best_model_path)
# import gym
# from gym_brt.envs import QubeSwingupEnv
# from gym_brt.control import QubeFlipUpControl

# with QubeSwingupEnv(use_simulator=False, frequency=250) as env:
#     controller = QubeFlipUpControl(sample_freq=250, env=env)
#     for episode in range(3):
#         state = env.reset()
#         state =np.array([np.cos(state[1]), np.sin(state[1]), state[3]], dtype=np.float32)
#         print(state.reshape(1,3))
#         for step in range(8000):
#             action, log_prob, value = policy.get_action(state.reshape(1,3))
#             # print(action[0][0])
#             state, reward, done, info = env.step(action[0][0])
#             state =np.array([np.cos(state[1]), np.sin(state[1]), state[3]], dtype=np.float32)

