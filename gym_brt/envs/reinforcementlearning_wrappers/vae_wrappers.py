import time

from gym import spaces
import numpy as np

from gym_brt.envs.reinforcementlearning_wrappers.rl_wrappers import FREQUENCY
from gym_brt.envs.reinforcementlearning_wrappers.vision_wrappers import VisionQubeBeginDownEnv
from visiontostate.signal_generator import SignalGenerator0upper
from visiontovision.vae_predictor import VAEPredictor, VAEPredictorSmall

"""
OpenAI Gym wrapper with feature vector of specified VAE model (data_id, model_name) as state
"""

real_state = True

class VAEQubeBeginDownEnv(VisionQubeBeginDownEnv):
    def __init__(self, data_id, model_name, frequency=FREQUENCY, batch_size=2048, use_simulator=False,
                 simulation_mode='ode', encoder_reset_steps=int(1e8), no_image_normalization=False):
        super().__init__(frequency, batch_size, use_simulator, simulation_mode, encoder_reset_steps,
                         no_image_normalization)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=[32], dtype=np.float32)

        # TODO prune network and just predict feature vector
        self.predictor = VAEPredictorSmall(data_id, model_name)

        if not real_state:
            self.goal_z = None
            self.goal_z = self.get_goal_state()

    def _get_state(self):
        image = super()._get_state()
        _, z = self.predictor.predict(image)
        self.z = z
        return z

    def _reward(self):
        if real_state:
            return super()._reward()
        if self.goal_z is None:
            self.goal_z = -1
            print('no reward returned')
            return 0.
        elif self.goal_z is -1:
            return 0.
        else:
            # 64 is max difference (range: -1,1 times 32)
            return 64-(np.square(self.z - self.goal_z)).mean()

    def get_goal_state(self):
        num_steps = self._frequency*5
        env = self
        featurelist = []
        print('Get goal state ... ')
        ctrl_sys = SignalGenerator0upper(env, sample_freq=self._frequency)

        env.reset()
        state, reward, done, info = env.step(np.array([0], dtype=np.float64))
        encoder_state = [info['params'], info['alpha'], info['theta_dot'], info['alpha_dot']]
        while ctrl_sys.step < ctrl_sys.sample_freq * (ctrl_sys.t_start + 5):
            # apply signal
            action = ctrl_sys.action(encoder_state)
            # get feedback
            state, reward, done, info = env.step(action)
            encoder_state = [info['params'], info['alpha'], info['theta_dot'], info['alpha_dot']]

            # # store data
            if ctrl_sys.step == ctrl_sys.sample_freq * ctrl_sys.t_start:
                print("Recording")
                featurelist = []
            if ctrl_sys.step >= ctrl_sys.sample_freq * ctrl_sys.t_start:
                featurelist.append(state)

        featurelist = np.array(featurelist)
        mean = featurelist.mean(axis=0)
        print('Goal state prediction done')
        print(mean)
        # TODO dispaly image with goal state and wait for q
        return mean