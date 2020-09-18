"""
Examples for calibrating the real Qube to a theta of 0.

@Author: Moritz Schneider
"""
from gym_brt.quanser import CalibrCtrl
from gym_brt.envs import QubeSwingupEnv
import math


def calibrate():
    frequency = 120
    u_max = 1.0
    desired_theta = (math.pi/180.) * 0.0

    with QubeSwingupEnv(frequency=frequency) as env:
        controller = CalibrCtrl(fs_ctrl=frequency, u_max=u_max, th_des=desired_theta)
        state = env.reset()
        while not controller.done:
            action = controller(state)
            state, reward, _, info = env.step(action)


if __name__ == '__main__':
    calibrate()
