import numpy as np
from gym_brt.envs import QubeBeginDownEnv, QubeSwingupEnv
from gym_brt.control import QubeFlipUpControl
import gym
import cv2
import time

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


class ImageObservationWrapper(gym.ObservationWrapper):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides
    """
    def __init__(self, env, out_shape=None):
        super(ImageObservationWrapper, self).__init__(env)
        dummy_obs = env.render("rgb_array")
        # Update observation space
        self.out_shape = out_shape

        obs_shape = out_shape if out_shape is not None else dummy_obs.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=dummy_obs.dtype)

    def observation(self, observation):
        #img = self.env.render("rgb_array")#, width=self.out_shape[0], height=self.out_shape[1])
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #if self.out_shape is not None:
        #    img = cv2.resize(img, (self.out_shape[0], self.out_shape[1]), interpolation=cv2.INTER_AREA)
        #cv2.imshow('image', img)
        img = self.env.render("rgb_array", width=self.out_shape[0], height=self.out_shape[1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imshow('image2', img2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return img

IMAGE_SHAPE = (220, 220, 3)

env = QubeSwingupEnv(frequency=100, use_simulator=True, simulation_mode='mujoco', integration_steps=1)
env.reward_range = (-float('inf'), float('inf'))

control = QubeFlipUpControl(env, sample_freq=100)


from gym import logger
logger.set_level(10)

#env.metadata.update(env.qube.metadata)
#env = ImageObservationWrapper(env, out_shape=IMAGE_SHAPE)
obs = env.reset()

pos = np.asarray([-np.pi/4., 2./3*np.pi])
vel = np.asarray([0, 0])
env.qube.set_state(pos, vel)
#env.qube.state = np.concatenate((pos, vel))
print(env.qube.state)
start = time.time()
for step in range(10):
    action = 0#control.action(obs)
    obs, reward, done, info = env.step(action)
    print(obs)
    env.render()
    #print(step)
    #env.render()
end = time.time()
print(end - start)
#cv2.destroyAllWindows()


