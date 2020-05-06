from gym_brt.envs.simulation.qube_simulation_base import QubeSimulatorBase
from gym_brt.envs.simulation.mujoco.mujoco_base import MujocoBase
import mujoco_py

import numpy as np

XML_PATH = "../../../data/xml/qube.xml"


class QubeMujoco(QubeSimulatorBase, MujocoBase):
    """Class for the Mujoco simulator."""

    def __init__(self, frequency=250, integration_steps=1, max_voltage=18.0):

        self._dt = 1.0 / frequency # TODO: See MujocoBase dt property
        self._integration_steps = integration_steps
        self._max_voltage = max_voltage

        MujocoBase.__init__(self, XML_PATH, 2)

        self.state = self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def step(self, action, led=None):
        action = np.clip(action, -self._max_voltage, self._max_voltage)
        self.do_simulation(action, self.frame_skip)
        self.state = self._get_obs()
        return self.state, {}

    def reset_up(self):
        # TODO
        self.state = (
            np.array([0, 0, 0, 0], dtype=np.float64) + np.random.randn(4) * 0.01
        )
        return self.state

    def reset_down(self):
        # TODO
        self.state = (
            np.array([0, np.pi, 0, 0], dtype=np.float64) + np.random.randn(4) * 0.01
        )
        return self.state

    def reset_encoders(self):
        pass
