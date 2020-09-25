"""
Examples for calibrating the real Qube to a theta of 0.

@Author: Moritz Schneider
"""
from gym_brt.control import calibrate
import math


def calibration():
    frequency = 120
    u_max = 1.0
    desired_theta = 0.0
    calibrate(desired_theta, frequency, u_max)


if __name__ == '__main__':
    calibration()
