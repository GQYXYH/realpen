from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy

from gym_brt.envs import QubeBeginUpEnv, QubeBeginDownEnv

MAX_EPISODE_LENGTH = 2000

# Parallel environments
env_kwargs = {'use_simulator': True, 'simulation_mode': 'mujoco', 'batch_size': MAX_EPISODE_LENGTH}
envs = make_vec_env(QubeBeginUpEnv, n_envs=1, env_kwargs=env_kwargs)

model = PPO(
    MlpPolicy,
    envs,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    ent_coef=0.0,
    learning_rate=2.5e-4,
    clip_range=0.2
)

model = model.load('models/qube_ppo_mujoco_2.zip')

env = QubeBeginDownEnv(use_simulator=True, simulation_mode='mujoco', batch_size=MAX_EPISODE_LENGTH)
for i in range(10):
    obs = env.reset()
    env.qube.randomise()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
