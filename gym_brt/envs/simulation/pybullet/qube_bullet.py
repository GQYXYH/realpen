"""PyBullet simulation for the Quanser QUBE-Servo 2.

This file was inspired from the following implementation:
https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/env_bases.py

The main difference between both files is, that this class does not inherit from the main environment class of OpenAI
Gym. This was done to avoid complications since the base class of the QUBE already inherits from 'gym.Env'.

Author: Moritz Schneider
"""

from gym_brt.envs.simulation.qube_simulation_base import QubeSimulatorBase
from pybullet_envs.robot_bases import MJCFBasedRobot, XmlBasedRobot
from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_utils import bullet_client

import gym
import pybullet as bullet
import numpy as np
import os

try:
  if os.environ["PYBULLET_EGL"]:
    import pkgutil
except:
  pass

XML_PATH = "../gym_brt/data/xml/qube_bullet.xml"


class QubeBullet(QubeSimulatorBase, XmlBasedRobot):
    """The Mujoco MJCF-based robot"""

    def __init__(self, frequency: float = 250, integration_steps: int = 1, max_voltage: float = 18.0):
        self.scene = None
        self.physicsClientId = -1
        self.isRender = True
        self.swingup = True
        self.seed()

        self.frequency = frequency
        self._dt = 1.0 / frequency # TODO: See MujocoBase dt property
        self._integration_steps = integration_steps
        self._max_voltage = max_voltage

        XmlBasedRobot.__init__(self, "qube", action_dim=1, obs_dim=4, self_collision=True)
        self.model_xml = XML_PATH
        self.doneLoading = 0
        MJCFBasedRobot.__init__(self, XML_PATH, "qube", action_dim=1, obs_dim=4)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        self.arm = self.parts["arm"]
        self.pen = self.parts["pole"]
        self.base_motor = self.jdict["base_motor"]
        self.arm_pendulum_joint = self.jdict["arm_pole"]

        u = self.np_random.uniform(low=-0.1, high=0.1)
        #self.base_motor.reset_current_position(u if not self.swingup else 3.1415 + u, 0)
        self.arm_pendulum_joint.reset_current_position(u if not self.swingup else 3.1415 + u, 0)

        self.base_motor.set_motor_torque(0)

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

    def angle_normalize(self, x: float) -> float:
        return (x % (2 * np.pi)) - np.pi

    def reset(self) -> np.ndarray:
        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

            if self.isRender:
                self._p = bullet_client.BulletClient(connection_mode=bullet.GUI)
            else:
                self._p = bullet_client.BulletClient()
            self._p.resetSimulation()
            self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
            #optionally enable EGL for faster headless rendering
            try:
                if os.environ["PYBULLET_EGL"]:
                    con_mode = self._p.getConnectionInfo()['connectionMethod']
                    if con_mode==self._p.DIRECT:
                        egl = pkgutil.get_loader('eglRenderer')
                        if (egl):
                            self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                        else:
                            self._p.loadPlugin("eglRendererPlugin")
            except:
                pass
            self.physicsClientId = self._p._client
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)

        if self.scene is None:
            self.scene = SingleRobotEmptyScene(self._p, gravity=9.81, timestep=self._dt, frame_skip=1)
        if not self.scene.multiplayer and self.ownsPhysicsClient:
            self.scene.episode_restart(self._p)

        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0

        #s = MJCFBasedRobot.reset(self, self._p)
        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                self.objects = self._p.loadMJCF(self.model_xml,
                                                flags=bullet.URDF_USE_SELF_COLLISION |
                                                      bullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                                      bullet.URDF_GOOGLEY_UNDEFINED_COLORS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(os.path.join(self.model_xml, flags=bullet.URDF_GOOGLEY_UNDEFINED_COLORS))
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
        self.robot_specific_reset(self._p)

        s = self._get_obs()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        self.potential = 0

        return s

    def reset_down(self) -> np.ndarray:
        self.reset()

    def reset_up(self) -> np.ndarray:
        self.swingup = False
        self.reset()

    def _get_obs(self) -> np.ndarray:
        theta, theta_dot = self.base_motor.current_position()
        alpha, alpha_dot = self.arm_pendulum_joint.current_position()
        #print("\n\nself.arm.current_position()", self.arm.current_position())
        #print("self.pen.current_position()", self.pen.current_position())

        return np.array([theta, alpha, theta_dot, alpha_dot])
    
    def step(self, action: float, led=None) -> np.ndarray:
        action = np.clip(action, -self._max_voltage, self._max_voltage)
        action = -self.gen_torque(action)
        self.base_motor.set_motor_torque(action)

        self.scene.global_step()
        state = self._get_obs()

        # TODO: ?
        #theta_x = state[0]
        #theta_y = state[1]
        #alpha_x = state[2]
        #alpha_y = state[3]
        #theta = np.arctan2(theta_y, theta_x)
        #alpha = np.arctan2(alpha_y, alpha_x)
        #theta_dot = state[4]
        #alpha_dot = state[5]


        #self.HUD(np.array([theta, alpha, theta_dot, alpha_dot]), action, False)
        return state
