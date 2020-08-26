import time

from gym import spaces
import numpy as np

from gym_brt.envs.reinforcementlearning_wrappers.rl_wrappers import FREQUENCY
from gym_brt.envs.reinforcementlearning_wrappers.vision_wrappers import VisionQubeBeginDownEnv
from visiontostate.signal_generator import SignalGenerator0upper
from visiontostate.vision_predictor import VisionPredictor
from visiontovision.vae_predictor import VAEPredictor, VAEPredictorSmall

"""
OpenAI Gym wrapper with predicted state vector of specified VisionToState model (data_id, model_name) as state
"""

class VtSQubeBeginDownEnv(VisionQubeBeginDownEnv):
    def __init__(self, data_id, model_name, frequency=FREQUENCY, batch_size=2048, use_simulator=False,
                 simulation_mode='ode', encoder_reset_steps=int(1e8), no_image_normalization=False):
        super().__init__(frequency, batch_size, use_simulator, simulation_mode, encoder_reset_steps,
                         no_image_normalization)

        self.predictor = VisionPredictor(data_id, model_name)

    def _get_state(self):
        image = super()._get_state()
        return self.predictor.predict(image)