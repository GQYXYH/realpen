# Extension modules for Reinforcement Learning

This directory includes methods, classes and wrapper to extend the normal Qube-Servo 2 classes especially for Reinforcement Learning.

- **[rl_gym_classes.py](./rl_gym_classes.py)**: Additional classes which can be used like the normal `QubeBaseEnv` classes (i.e. `QubeSwingupEnv`).
- **[rl_reward_functions.py](./rl_reward_functions.py)**: Additional reward functions for training. To use them, the reward function of the used Gym class must be overwritten or changed by an own wrapper.
- **[wrapper.py](./wrapper.py)**: Wrapper to enforce different behaviors of the used Gym class (i.e. different reward function, image-like observation instead of low-dimensional states, etc.). Some of the concepts are the same as in *[rl_gym_classes.py](./rl_gym_classes.py)* and *[rl_reward_functions.py](./rl_reward_functions.py)* but in additional wrapper form. Wrapper can be used like `env = wrapper_cls(env)` and furthermore can be nested. [More on wrappers here](www.github.com/openai/gym/blob/master/gym/core.py).