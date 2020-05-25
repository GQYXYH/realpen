import mujoco_py
import numpy as np
from gym_brt.envs import QubeSwingupEnv, QubeBeginUpEnv, QubeBeginDownEnv

xml_path = "../gym_brt/data/xml/qube.xml"
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

viewer = mujoco_py.MjViewer(sim)

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


env = QubeBeginDownEnv(use_simulator=True, simulation_mode='mujoco')
obs = env.reset()
agent = ChangingAgent()
while True:
    action = agent.step()
    obs, r, done, _ = env.step(action)
    done = done.any() if isinstance(done, np.ndarray) else done
    env.render()

    if done:
        obs = env.reset()
        #env.qube.randomise()
