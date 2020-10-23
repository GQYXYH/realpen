"""
Examples for interacting with the OpenAI gym like environements.

Descriptions of the environemts can be found in in the classes itself.
Energy Based Swing Up and LQR controller are implemented in QubeFlipUpControl.

Reward can be defined under gym_brt/envs/reinforcementlearning_extensions/rl_reward_functions.py
"""
from gym_brt.control import QubeFlipUpControl, QubeHoldControl, calibrate
from gym_brt.envs import QubeSwingupEnv, QubeBalanceEnv


def interact_with_down_environment(n_trials: int = 1):
    assert n_trials >= 1
    frequency = 120
    calibrate(0.0)

    with QubeSwingupEnv(frequency=frequency) as env:
        controller = QubeFlipUpControl(sample_freq=frequency)
        for episode in range(n_trials):
            state = env.reset()
            for step in range(100000):
                action = controller.action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break


def interact_with_balance_env(n_trials: int = 1):
    assert n_trials >= 1
    frequency = 150

    with QubeBalanceEnv(use_simulator=False, frequency=frequency) as env:
        controller = QubeHoldControl(sample_freq=frequency, env=env)
        for episode in range(n_trials):
            state = env.reset()
            for step in range(30000):
                action = controller.action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break


if __name__ == '__main__':
    interact_with_down_environment(n_trials=1)
