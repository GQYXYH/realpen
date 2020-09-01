from gym_brt.envs.simulation.qube_simulation_base import QubeSimulatorBase
from gym_brt.envs.simulation.mujoco.mujoco_base import MujocoBase
import mujoco_py
from mujoco_py.modder import TextureModder

import numpy as np

#XML_PATH = "../gym_brt/data/xml/qube.xml"
XML_PATH = "qube.xml"


class QubeMujoco(QubeSimulatorBase, MujocoBase):
    """Class for the Mujoco simulator."""

    def reset(self):
        MujocoBase.reset(self)

    def __init__(
            self,
            frequency: float = 250,
            integration_steps: int = 1,
            max_voltage: float = 18.0
        ):
        self._dt = 1.0 / frequency # TODO: See MujocoBase dt property
        self._integration_steps = integration_steps
        self._max_voltage = max_voltage

        self.Rm = 8.4  # Resistance
        self.kt = 0.042  # Current-torque (N-m/A)
        self.km = 0.042  # 0.042  # Back-emf constant (V-s/rad)

        MujocoBase.__init__(self, XML_PATH, integration_steps)
        self.model.opt.timestep = self._dt
        #self.frame_skip = int(1 / frequency * self.model.opt.timestep)

        self.state = self._get_obs()

    def _get_obs(self):
        """
        qpos: params, alpha
        qvel: theta_dot, alpha_dot
        :return: Numpy array of the form: [params alpha theta_dot alpha_dot]
        """
        theta_before, alpha_before = self.sim.data.qpos
        theta_dot, alpha_dot = self.sim.data.qvel

        theta = self.angle_normalize(theta_before + np.pi)
        alpha = self.angle_normalize(alpha_before + np.pi)

        #alpha_dot *= -1

        return -np.array([theta, alpha, theta_dot, alpha_dot])

    def angle_normalize(self, x: float) -> float:
        return (x % (2 * np.pi)) - np.pi

    def gen_torque(self, action) -> float:
        # Motor
        # Rotor inertia 4e-06

        _, _, theta_dot, _ = self._get_obs()

        tau = -(self.kt * (action - self.km * theta_dot)) / self.Rm  # torque
        return tau

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def step(self, action: float, led=None) -> np.array:
        action = np.clip(action, -self._max_voltage, self._max_voltage)
        action = -self.gen_torque(action)
        self.do_simulation(action)
        self.state = self._get_obs()
        return self.state

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_up(self):
        qpos = np.array([0, 0], dtype=np.float64) + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = np.array([0, 0], dtype=np.float64) + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_down(self):
        qpos = np.array([0, np.pi], dtype=np.float64) + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = np.array([0, 0], dtype=np.float64) + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_encoders(self):
        pass
