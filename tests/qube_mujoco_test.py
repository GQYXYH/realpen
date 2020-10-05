import numpy as np
from gym_brt.envs import QubeBeginDownEnv

from simmod.modification.mujoco import MujocoTextureModifier, MujocoMaterialModifier
from simmod.wrappers import UDRMujocoWrapper
from gym.wrappers import Monitor

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

## Create the needed modifiers
#tex_mod = MujocoTextureModifier(env.qube.sim)
#mat_mod = MujocoMaterialModifier(env.qube.sim)

def video_callable(episode_id):
    return True

from gym import logger
logger.set_level(10)

env.metadata.update(env.qube.metadata)
# Wrap the environment
#env = UDRMujocoWrapper(env, tex_mod, mat_mod, sim=env.qube.sim)
env = Monitor(env, directory="./monitor", video_callable=video_callable, force=True)


obs = env.reset()

#from PIL import Image
#img = Image.fromarray(obs['pixels'], 'RGB')
#img.show()

for step in range(5):
    action = 0
    #if step % 10 == 0:
    #    env.alg.step()
    obs, reward, done, info = env.step(action)
    #env.render()

    #img = Image.fromarray(obs['pixels'], 'RGB')
    #img.show()

