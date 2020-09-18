from gym_brt.control import QubeFlipUpControl
from gym_brt.control.control import QubeHoldControl
from gym_brt.envs import QubeSwingupEnv, QubeBalanceEnv


"""
Examples for interacting with the OpenAI gym like environements. 

Descriptions of the environemts can be found in in the classes itself.
Energy Based Swing Up and LQR controller are implemented in QubeFlipUpControl. 

Reward can be defined under gym_brt/envs/reinforcementlearning_wrappers/rl_reward_functions.py
"""


def interact_with_down_environment():
    frequency = 120

    with QubeSwingupEnv(frequency=frequency) as env:
        controller = QubeFlipUpControl(sample_freq=frequency)
        for episode in range(11):
            ret = 0
            state = env.reset()
            for step in range(30000):
                action = controller.action(state)
                state, reward, done, info = env.step(action)
                ret += reward
                if done:
                    print(ret)
                    break


def interact_with_balance_env():
    frequency = 150

    with QubeBalanceEnv(use_simulator=False, frequency=frequency) as env:
        controller = QubeHoldControl(sample_freq=frequency, env=env)
        for episode in range(2):
            state = env.reset()
            for step in range(30000):
                action = controller.action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break

if __name__ == '__main__':
    interact_with_down_environment()
    #interact_with_balance_env()