from gym_brt.envs.simulation.qube_simulation_base import QubeSimulatorBase
from pybulletgym.envs.mujoco.scene_bases import SingleRobotEmptyScene
from pybulletgym.envs.mujoco.robot_bases import MJCFBasedRobot
from pybulletgym.envs.mujoco.env_bases import BaseBulletEnv
import numpy as np
import os

XML_PATH = "../gym_brt/data/xml/qube_original.xml"


class QubeBulletRobot(MJCFBasedRobot):
    """The Mujoco MJCF-based robot"""

    def __init__(self, swingup=True):
        self.swingup = swingup
        dir_path = os.path.dirname(os.path.realpath(__file__))
        MJCFBasedRobot.__init__(
            self, XML_PATH, "cart", action_dim=1, obs_dim=4
        )

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.arm = self.parts["arm"]
        self.pen = self.parts["pole"]
        self.rotary_top = self.jdict["base_motor"]
        self.arm_pendulum_hinge = self.jdict["arm_pole"]

        u = self.np_random.uniform(low=-0.1, high=0.1)
        self.rotary_top.reset_current_position(u if not self.swingup else 3.1415 + u, 0)

        self.rotary_top.set_motor_torque(0)

    def gen_torque(self, action) -> float:
        # Motor
        Rm = 8.4  # Resistance
        kt = 0.042  # Current-torque (N-m/A)
        km = 0.042  # 0.042  # Back-emf constant (V-s/rad)
        # Rotor inertia 4e-06

        #Rm, kt, km = self.actuation_consts

        _, _, theta_dot, _ = self._get_obs()

        Vm = action
        tau = -(kt * (Vm - km * theta_dot)) / Rm  # torque
        return tau

    def apply_action(self, a):
        self.rotary_top.set_motor_torque(a)

    def calc_state(self):

        t_, theta, theta_dot = self.arm.current_position()
        a_, alpha, alpha_dot = self.pen.current_position()
        print("\n\nself.arm.current_position()", self.arm.current_position())
        print("self.pen.current_position()", self.pen.current_position())

        # Equivalents in Mujoco
        # qpos == np.array([params, alpha]) == self.sim.data.qpos
        # qvel == np.array([theta_dot, alpha_dot]) == self.sim.data.qvel
        return np.array(
            [
                np.cos(theta),
                np.sin(theta),
                np.cos(alpha),
                np.sin(alpha),
                theta_dot,
                alpha_dot,
            ]
        )


class QubeBullet(QubeSimulatorBase, BaseBulletEnv):
    """The environment that wraps the robot"""

    def __init__(self,
                 frequency: float = 250,
                 integration_steps: int = 1,
                 max_voltage: float = 18.0
                 ):
        self.robot = QubeBulletRobot(swingup=True)
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1
        self.frequency = frequency

        self._dt = 1.0 / frequency # TODO: See MujocoBase dt property
        self._integration_steps = integration_steps
        self._max_voltage = max_voltage

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(
            bullet_client, gravity=9.81, timestep=self._dt, frame_skip=1
        )

    def angle_normalize(self, x: float) -> float:
        return (x % (2 * np.pi)) - np.pi

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        r = BaseBulletEnv._reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()

        return r

    def reset_down(self):
        pass

    def reset_up(self):
        pass

    def step(self, action: float, led=None):
        self.robot.apply_action(action)
        self.scene.global_step()
        state = self.robot.calc_state()

        theta_x = state[0]
        theta_y = state[1]
        alpha_x = state[2]
        alpha_y = state[3]
        theta = np.arctan2(theta_y, theta_x)
        alpha = np.arctan2(alpha_y, alpha_x)
        theta_dot = state[4]
        alpha_dot = state[5]

        reward = state[2] - 0.01 * np.abs(theta)

        self.HUD(np.array([theta, alpha, theta_dot, alpha_dot]), a, False)
        return state, reward, False, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0, 1.2, 1.2, 0, 0, 0.5)