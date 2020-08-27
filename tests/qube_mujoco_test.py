import mujoco_py
import numpy as np
from gym_brt.envs import QubeSwingupEnv, QubeBeginUpEnv, QubeBeginDownEnv
from gym_brt.envs.simulation.mujoco import QubeMujoco

from gym.wrappers.pixel_observation import PixelObservationWrapper

class ChangingAgent():

    def __init__(self, steps=20, action_value=7):
        self._changing_steps = steps
        self._action_value = action_value
        self.counter = 0
        self.direction = 1

    def step(self):
        if self.counter % self._changing_steps == 0:
            self.direction *= -1
        self.counter += 1
        return self.direction * self._action_value

# Square wave, switch every 85 ms
def square_wave_policy(state, step, frequency=250, **kwargs):
    # steps_until_85ms = int(85 * (frequency / 300))
    # state = _convert_state(state)
    # # Switch between positive and negative every 85 ms
    # mod_170ms = step % (2 * steps_until_85ms)
    # if mod_170ms < steps_until_85ms:
    #     action = 3.0
    # else:
    #     action = -3.0
    action = 3.0*np.sin(step/frequency/0.1)

    return np.array([action])


def set_init_from_ob(env, ob):
    pos = ob[:2]
    pos[-1] *= -1
    vel = ob[2:]
    vel[-1] *= -1

    env.set_state(pos, vel)
    return env._get_obs()

env = QubeBeginDownEnv(frequency=100, use_simulator=True, simulation_mode='mujoco')
env.reward_range = (-float('inf'), float('inf'))
env = PixelObservationWrapper(env)
obs = env.reset()

#from PIL import Image
#img = Image.fromarray(obs['pixels'], 'RGB')
#img.show()

for step in range(100):
    action = 0
    obs, reward, done, info = env.step(action)
    env.render()

    #img = Image.fromarray(obs['pixels'], 'RGB')
    #img.show()

