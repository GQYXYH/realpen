from gym_brt.envs.reinforcementlearning_wrappers.rl_wrappers import FREQUENCY
from gym_brt.envs.reinforcementlearning_wrappers.vision_wrappers import VisionQubeBeginDownEnv
from visiontovision.vae_predictor import VAEPredictor

"""
OpenAI Gym wrapper with feature vector of specified VAE model (data_id, model_name) as state
"""


class VAEQubeBeginDownEnv(VisionQubeBeginDownEnv):
    def __init__(self, data_id, model_name, frequency=FREQUENCY, batch_size=2048, use_simulator=False,
                 simulation_mode='ode', encoder_reset_steps=int(1e8), no_image_normalization=False):
        super().__init__(frequency, batch_size, use_simulator, simulation_mode, encoder_reset_steps,
                         no_image_normalization)

        # TODO prune network and just predict feature vector
        self.predictor = VAEPredictor(data_id, model_name)

    def _get_state(self):
        image = super()._get_state()
        _, z = self.predictor.predict(image)
        return z