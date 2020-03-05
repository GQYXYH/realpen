import numpy as np
from gym import spaces
from gym_brt.envs.reinforcementlearning_wrappers.rl_wrappers import FREQUENCY

from blackfly.blackfly import Blackfly
from blackfly.image_preprocessor import ImagePreprocessor, IMAGE_SHAPE
from gym_brt.envs import QubeBeginDownEnv, QubeBeginUpEnv
from gym_brt.envs.qube_base_env import ACT_MAX

"""Wrapper classes for the QubeEnv of the quanser driver. The wrappers work as an OpenAi Gym interface. The state 
itself is the image grabbed by the camera, in info['state'] the states from the encoders can be found. It 
just contains the states [cos(theta), sin(theta), cos(alpha), sin(alpha), theta_velocity, alpha_velocity]. The reward 
functions provided can be found in rl_reward_functions.py. 

Wrapper always needs to be used like
    with Wrapper as wrapper:
to ensure safe closure of camera and qube!
"""


class SwingUpVisionWrapper(QubeBeginDownEnv):

    def __init__(self, preprocess=False, frequency=FREQUENCY, image_shape=IMAGE_SHAPE):
        super().__init__(
            frequency=frequency,
            use_simulator=False)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=IMAGE_SHAPE, dtype=np.float32)
        self.action_space = spaces.Box(-ACT_MAX, ACT_MAX, dtype=np.float32)

        self.camera = Blackfly(exposure_time=1000)
        self.camera.start_acquisition()
        self.preprocessor = ImagePreprocessor(preprocess=preprocess, image_shape=image_shape)
        # warm up
        image = self.camera.get_image()
        self.preprocessor.preprocess_and_normalize_image(image)

    def __exit__(self, typeoferror, value, traceback):
        # safely close qube
        super().__exit__(typeoferror, value, traceback)
        # safely end aquisition and close camera
        self.camera.end_acquisition()
        self.camera.__exit__(typeoferror, value, traceback)

    def step(self, action):
        state, reward, _, info = super().step(action)
        done = self._isdone()
        image = self.camera.get_image()
        try:
            image = self.preprocessor.preprocess_and_normalize_image(image)
        except:
            print('got empty image!! (in step())')
            self.__exit__(None, None, None)
        info['state'] = state[0:6]
        return image, reward, done, info

    def reset(self):
        # Start the pendulum stationary at the bottom (stable point)
        super().reset()
        image = self.camera.get_image()
        try:
            image = self.preprocessor.preprocess_and_normalize_image(image)
        except:
            print('got empty image!! (in reset())')
            self.__exit__(None, None, None)
        return image


class BalanceVisionWrapper(QubeBeginUpEnv):

    def __init__(self, preprocess=False, frequency=FREQUENCY, image_shape=IMAGE_SHAPE):
        super().__init__(
            frequency=frequency,
            use_simulator=False)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=IMAGE_SHAPE, dtype=np.float32)
        self.action_space = spaces.Box(-ACT_MAX, ACT_MAX, dtype=np.float32)

        self.camera = Blackfly(exposure_time=1000)
        self.camera.start_acquisition()
        self.preprocessor = ImagePreprocessor(preprocess=preprocess, image_shape=image_shape)
        # warm up
        image = self.camera.get_image()
        self.preprocessor.preprocess_and_normalize_image(image)

    def __exit__(self, typeoferror, value, traceback):
        # safely close qube
        super().__exit__(typeoferror, value, traceback)
        # safely end aquisition and close camera
        try:
            self.camera.end_acquisition()
        except:
            print('could not end acquisition')
        try:
            self.camera.__exit__(typeoferror, value, traceback)
        except:
            print("could not exit camera")

    def step(self, action):
        state, reward, _, info = super().step(action)
        done = self._isdone()
        image = self.camera.get_image()
        try:
            image = self.preprocessor.preprocess_and_normalize_image(image)
        except:
            print('got empty image!! (in reset())')
            self.__exit__(None, None, None)
        info['state'] = state[0:6]
        return image, reward, done, info

    def reset(self):
        # Start the pendulum stationary at the top (stable point)
        super().reset()
        image = self.camera.get_image()
        try:
            image = self.preprocessor.preprocess_and_normalize_image(image)
        except:
            print('got empty image!! (in step())')
            self.__exit__(None, None, None)
        return image
