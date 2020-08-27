import numpy as np
import cv2

from blackfly.blackfly import Blackfly
from blackfly.image_preprocessor import IMAGE_SHAPE, ImagePreprocessor
from gym_brt.envs.qube_swingup_env import QubeSwingupEnv
from gym_brt.envs.reinforcementlearning_wrappers.rl_wrappers import FREQUENCY

from gym import ObservationWrapper, spaces

"""
Wrapper classes for the QubeEnv of the quanser driver. The wrappers work as an OpenAi Gym interface. 

These wrappers can be used to get consistent environments with the following state representation:
[cos(theta), sin(theta), cos(alpha), sin(alpha), theta_dot, alpha_dot]. 

The reward functions provided can be found in rl_reward_functions.py. 

Wrapper always needs to be used like
    with Wrapper as wrapper:
to ensure safe closure of camera and qube!
"""


class ImageObservationWrapper(ObservationWrapper):
    """
    Use env.render(rgb_array) as observation
    rather than the observation the environment provides
    """
    def __init__(self, env, out_shape=None):
        super(ImageObservationWrapper, self).__init__(env)
        dummy_obs = env.render("rgb_array")
        # Update observation space
        self.out_shape = out_shape

        obs_shape = out_shape if out_shape is not None else dummy_obs.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=dummy_obs.dtype)

    def observation(self, observation):
        img = self.env.render("rgb_array")
        if self.out_shape is not None:
            img = cv2.resize(img, (self.out_shape[0], self.out_shape[1]), interpolation=cv2.INTER_AREA)
        return img


class BlackFlyWrapper(ObservationWrapper):
    """
    Use images from a BlackFly camera as observation
    rather than the observation the environment provides
    """
    def __init__(self, env, no_image_normalization=False):
        super(BlackFlyWrapper, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=IMAGE_SHAPE, dtype=np.float32)
        self.camera = Blackfly(exposure_time=1000)
        self.camera.start_acquisition()
        self.preprocessor = ImagePreprocessor(False, IMAGE_SHAPE)
        self.no_image_normalization = no_image_normalization

    def _get_state(self):
        image = self.camera.get_image()
        if self.no_image_normalization:
            return self.preprocessor.preprocess_image(image)
        else:
            return self.preprocessor.preprocess_and_normalize_image(image)

    def __enter__(self):
        print('start camera')
        self.camera.start_acquisition()
        return super().__enter__()

    def __exit__(self, type, value, traceback):
        try:
            self.camera.end_acquisition()
        except:
            print('could not end camera')
            pass
        self.camera.__exit__(type, value, traceback)
        super().__exit__(type, value, traceback)

    def observation(self, observation):
        return self._get_state()


class VisionQubeBeginDownEnv(QubeSwingupEnv):
    def __init__(self, frequency=FREQUENCY, batch_size=2048, use_simulator=False, simulation_mode='ode',
                 encoder_reset_steps=int(1e8), no_image_normalization=False):
        super().__init__(frequency, batch_size, use_simulator, simulation_mode, encoder_reset_steps, )
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=IMAGE_SHAPE, dtype=np.float32)

        if use_simulator:
            if simulation_mode == 'mujoco':
                # TODO Moritz, add code for mujoco
                pass
            else:
                raise ValueError(f"Unsupported simulation type '{simulation_mode}'. "
                                 f"Valid ones are 'mujoco'")
        else:
            self.camera = Blackfly(exposure_time=1000)
            self.camera.start_acquisition()
            self.preprocessor = ImagePreprocessor(False, IMAGE_SHAPE)

        self.use_simulator = use_simulator
        self.simulation_mode = simulation_mode
        self.no_image_normalization = no_image_normalization

    def _get_state(self):
        if self.use_simulator:
            if self.simulation_mode == 'mujoco':
                # TODO Moritz, add code for mujoco
                pass
            else:
                raise ValueError(f"Unsupported simulation type '{self.simulation_mode}'. "
                                 f"Valid ones are 'mujoco'")
        else:
            image = self.camera.get_image()
            if self.no_image_normalization:
                return self.preprocessor.preprocess_image(image)
            else:
                return self.preprocessor.preprocess_and_normalize_image(image)

    def __enter__(self):
        print('start camera')
        # self.camera.start_acquisition()
        return super().__enter__()

    def __exit__(self, type, value, traceback):
        try:
            self.camera.end_acquisition()
        except:
            print('could not end camera')
            pass
        self.camera.__exit__(type, value, traceback)
        super().__exit__(type, value, traceback)