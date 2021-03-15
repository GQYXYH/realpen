"""
OpenAI Gym environments with predicted state vector of specified VisionToState model (data_id, model_name) as state

@Author: Steffen Bleher
"""
from gym import spaces
from gym_brt.data.config.configuration import FREQUENCY

import numpy as np

from rl_classes.vision_wrapping_classes import VisionQubeBeginDownEnv
from visiontostate.vision_predictor import VisionPredictor

OBS_MAX = np.asarray([1, 1, 1, 1, np.inf, np.inf], dtype=np.float64)


class VtSQubeBeginDownEnv(VisionQubeBeginDownEnv):
    def __init__(self, data_id, model_name, frequency=FREQUENCY, batch_size=2048, use_simulator=False,
                 simulation_mode='ode', encoder_reset_steps=int(1e8), no_image_normalization=False):
        super().__init__(frequency, batch_size, use_simulator, simulation_mode, encoder_reset_steps,
                         no_image_normalization)

        self.observation_space = spaces.Box(-OBS_MAX, OBS_MAX, dtype=np.float32)
        self.predictor = VisionPredictor(data_id, model_name)

    def _get_state(self):
        image = super()._get_state()
        state = self.predictor.predict(image)
        return state[0:6]