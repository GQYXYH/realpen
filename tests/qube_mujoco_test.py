import mujoco_py
import numpy as np
from gym_brt.envs import QubeSwingupEnv, QubeBeginUpEnv, QubeBeginDownEnv
from gym_brt.envs.simulation.mujoco import QubeMujoco

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


env = QubeSwingupEnv(use_simulator=True, simulation_mode='mujoco')
obs = env.reset()
t = 0
while True:
    action = square_wave_policy(obs, t)
    obs, r, done, _ = env.step(action)
    print(f"Step {t}: \t {obs}, \t \t Action: \t {action}")
    done = done.any() if isinstance(done, np.ndarray) else done
    env.render()

    if t == 1000:
        break
    t += 1
