"""
Examples for calibrating the real Qube to a theta of 0.

@Author: Moritz Schneider
"""
from gym_brt.control import calibrate, QubeFlipUpControl
from gym_brt.envs.reinforcementlearning_extensions.wrapper import CalibrationWrapper
from gym_brt.envs import QubeSwingupEnv
import math


def test_calibration():
    frequency = 120
    u_max = 1.0
    desired_theta = 0.0
    calibrate(desired_theta=desired_theta, frequency=frequency, u_max=u_max)


def test_calibration_wrapper():
    n_trials = 3
    frequency = 120

    with CalibrationWrapper(QubeSwingupEnv(frequency=frequency), noise=True) as env:
        controller = QubeFlipUpControl(sample_freq=frequency)
        for episode in range(n_trials):
            state = env.reset()
            for step in range(30000):
                action = controller.action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break


if __name__ == '__main__':
    test_calibration_wrapper()
