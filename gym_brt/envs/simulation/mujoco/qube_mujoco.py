"""Mujoco simulation for the Quanser QUBE-Servo 2.

@Author: Moritz Schneider
"""

import numpy as np

from gym_brt.envs.simulation.mujoco.mujoco_base import MujocoBase
from gym_brt.envs.simulation.qube_simulation_base import QubeSimulatorBase

# XML_PATH = "../gym_brt/data/xml/qube.xml"
XML_PATH = "qube.xml"


def _normalize_angle(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class QubeMujoco(QubeSimulatorBase, MujocoBase):
    """Class for the Mujoco simulator."""

    def __init__(self, frequency: float = 250, integration_steps: int = 1, max_voltage: float = 18.0):
        """Creates the Mujoco simulation of the Qube-Servo 2.

        Args:
            frequency: Frequency of the simulation
            integration_steps: Number of integration steps tp be made during a single simulation step
            max_voltage: Maximum voltage which can be applied
        """
        self._dt = 1.0 / frequency
        self._integration_steps = integration_steps
        self._max_voltage = max_voltage

        self.Rm = 8.9 #8.4  # Resistance
        self.kt = 0.035 #0.035 #0.042 # Current-torque (N-m/A)
        self.km = 0.043 #0.035 # Back-emf constant (V-s/rad)

        MujocoBase.__init__(self, XML_PATH, integration_steps)
        self.model.opt.timestep = self._dt
        # self.frame_skip = int(1 / frequency * self.model.opt.timestep)
        self.state = self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Calculate normalized observations from the current simulation state.

        Returns:
            Numpy array of the form: [theta alpha theta_dot alpha_dot]
        """
        theta_before, alpha_before = self.sim.data.qpos
        theta_dot, alpha_dot = self.sim.data.qvel

        theta = _normalize_angle(theta_before)
        alpha = _normalize_angle(alpha_before)

        return -np.array([theta, alpha, theta_dot, alpha_dot]) # TODO: Remove -1

    def gen_torque(self, action) -> float:
        _, _, theta_dot, _ = self._get_obs()

        tau = -(self.kt * (action - self.km * theta_dot)) / self.Rm  # torque
        return tau

    def viewer_setup(self) -> None:
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def step(self, action: float, led=None) -> np.array:
        action = np.clip(action, -self._max_voltage, self._max_voltage)
        action = -self.gen_torque(action) #TODO: Remove -1
        self.do_simulation(action)
        self.state = self._get_obs()
        return self.state

    def reset(self):
        MujocoBase.reset(self)

    def reset_model(self) -> np.ndarray:
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel #+ self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_up(self) -> np.ndarray:
        qpos = np.array([0, 0], dtype=np.float64) #+ self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = np.array([0, 0], dtype=np.float64) #+ self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_down(self) -> np.ndarray:
        qpos = np.array([0, np.pi], dtype=np.float64) #+ self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = np.array([0, 0], dtype=np.float64) #+ self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_encoders(self):
        pass
