import functools

import matplotlib.pyplot as plt

import numpy as np
from gym_brt.envs.reinforcementlearning_wrappers.rl_wrappers import FREQUENCY

from gym_brt.quanser import QubeHardware, QubeSimulator
from gym_brt.quanser.qube_interfaces import forward_model_ode

from gym_brt.control.control import _convert_state, QubeFlipUpControl


# No input
from simulator_tuning.qube_simulator_optimize import forward_model_ode_optimize


def zero_policy(state, **kwargs):
    return np.array([0.0])


# Constant input
def constant_policy(state, **kwargs):
    return np.array([3.0])


# Rand input
def random_policy(state, **kwargs):
    return np.asarray([np.random.randn()])


# Square wave, switch every 85 ms
def square_wave_policy(state, step, frequency=FREQUENCY, **kwargs):
    # steps_until_85ms = int(85 * (frequency / 300))
    # state = _convert_state(state)
    # # Switch between positive and negative every 85 ms
    # mod_170ms = step % (2 * steps_until_85ms)
    # if mod_170ms < steps_until_85ms:
    #     action = 3.0
    # else:
    #     action = -3.0
    action = 3.0*np.sin(step/frequency/0.1)

    return np.array([action])


# Flip policy
def energy_control_policy(state, **kwargs):
    state = _convert_state(state)
    # Run energy-based control to flip up the pendulum
    theta, alpha, theta_dot, alpha_dot = state
    # alpha_dot += alpha_dot + 1e-15

    """Implements a energy based swing-up controller"""
    mu = 50.0  # in m/s/J
    ref_energy = 30.0 / 1000.0  # Er in joules

    # TODO: Which one is correct?
    max_u = 6  # Max action is 6m/s^2
    # max_u = 0.85  # Max action is 6m/s^2

    # System parameters
    jp = 3.3282e-5
    lp = 0.129
    lr = 0.085
    mp = 0.024
    mr = 0.095
    rm = 8.4
    g = 9.81
    kt = 0.042

    pend_torque = (1 / 2) * mp * g * lp * (1 + np.cos(alpha))
    energy = pend_torque + (jp / 2.0) * alpha_dot * alpha_dot

    u = mu * (energy - ref_energy) * np.sign(-1 * np.cos(alpha) * alpha_dot)
    u = np.clip(u, -max_u, max_u)

    torque = (mr * lr) * u
    voltage = (rm / kt) * torque
    return np.array([-voltage])


# Hold policy
def pd_control_policy(state, **kwargs):
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state
    # multiply by proportional and derivative gains
    kp_theta = -2.0
    kp_alpha = 35.0
    kd_theta = -1.5
    kd_alpha = 3.0

    # If pendulum is within 20 degrees of upright, enable balance control, else zero
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = (
                theta * kp_theta
                + alpha * kp_alpha
                + theta_dot * kd_theta
                + alpha_dot * kd_alpha
        )
    else:
        action = 0.0
    action = np.clip(action, -3.0, 3.0)
    return np.array([action])


# Flip and Hold
def flip_and_hold_policy(state, **kwargs):
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    # If pendulum is within 20 degrees of upright, enable balance control
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = pd_control_policy(state)
    else:
        action = energy_control_policy(state)
    return action


# Square wave instead of energy controller flip and hold
def square_wave_flip_and_hold_policy(state, **kwargs):
    state = _convert_state(state)
    theta, alpha, theta_dot, alpha_dot = state

    # If pendulum is within 20 degrees of upright, enable balance control
    if np.abs(alpha) <= (20.0 * np.pi / 180.0):
        action = pd_control_policy(state)
    else:
        action = square_wave_policy(state, **kwargs)
    return action


# Run on the hardware
def run_qube(begin_up, policy, nsteps, frequency, integration_steps):
    with QubeHardware(frequency=frequency, max_voltage=3.0) as qube:
        if begin_up is True:
            s = qube.reset_up()
        elif begin_up is False:
            s = qube.reset_down()

        # Wait for a little bit before reading the initial state
        for _ in range(15):
            a = random_policy(s)
            s = qube.step(a)

        init_state = s
        a = policy(s)
        s_hist = [s]
        a_hist = [a]

        for i in range(nsteps):
            s = qube.step(a)
            a = policy(s)

            s_hist.append(s)  # States
            a_hist.append(a)  # Actions

        # Return a 2d array, hist[n,d] gives the nth timestep and the dth dimension
        # Dims are ordered as: ['Theta', 'Alpha', 'Theta dot', 'Alpha dot', 'Action']
        return np.concatenate((np.array(s_hist), np.array(a_hist)), axis=1), init_state


def run_sim(init_state, policy, nsteps, frequency, integration_steps, params=None):
    if params is not None:
        sim_function = functools.partial(forward_model_ode_optimize, params=params)
    else:
        sim_function = forward_model_ode

    with QubeSimulator(
            forward_model=sim_function,
            frequency=frequency,
            integration_steps=integration_steps,
            max_voltage=3.0,
    ) as qube:
        qube.state = np.asarray(init_state, dtype=np.float64)  # Set the initial state of the simulator

        s = init_state
        a = policy(s, step=0)
        s_hist = [s]
        a_hist = [a]

        for i in range(nsteps):
            s = qube.step(a)
            a = policy(s, step=i + 1, frequency=frequency)

            s_hist.append(s)  # States
            a_hist.append(a)  # Actions

        # Return a 2d array, hist[n,d] gives the nth timestep and the dth dimension
        # Dims are ordered as: ['Theta', 'Alpha', 'Theta dot', 'Alpha dot', 'Action']
        return np.concatenate((np.array(s_hist), np.array(a_hist)), axis=1)


def plot_results(hists, labels, colors=None, normalize=None):
    state_dims = ['Theta', 'Alpha', 'Theta dot', 'Alpha dot', 'Action']

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    for i, ax in enumerate((ax1, ax2, ax3, ax4, ax5)):

        # Normalize to be between 0 and 2pi
        if normalize is not None:
            if state_dims[i].lower() in normalize:
                for hist in hists:
                    hist[:, i] %= 2 * np.pi

        # Plot with specific colors
        if colors is not None:
            for hist, label, color in zip(hists, labels, colors):
                ax.plot(hist[:, i], label=label, color=color)

        # Default colors
        else:
            for hist, label in zip(hists, labels):
                ax.plot(hist[:, i], label=label)
        ax.set_ylabel(state_dims[i])
        ax.legend()
    plt.show()


def run_real():
    # Natural response when starting at α = 0 + noise (upright/inverted)
    hist_qube, init_state = run_qube(True, flip_and_hold_policy, nsteps, frequency, i_steps)

    import pickle
    outfile = open("../hist_qube", "wb")
    pickle.dump(hist_qube, outfile)
    outfile = open("../init_state", "wb")
    pickle.dump(init_state, outfile)


def run(params=None, visualize=False):
    import pickle
    infile = open("../hist_qube", "rb")
    hist_qube = pickle.load(infile)
    infile = open("../init_state", "rb")
    init_state = pickle.load(infile)
    hist_ode = run_sim(init_state, flip_and_hold_policy, nsteps, frequency, i_steps, params=params)
    print(params)

    if visualize:
        # plot_results(hists=[hist_qube], labels=['Hardware'], colors=None)
        plot_results(hists=[hist_qube, hist_ode], labels=['Hardware', 'ODE'], normalize=['alpha'])
    return np.sqrt(np.mean((hist_qube[:,0] - hist_ode[:,0]) ** 2))


if __name__ == '__main__':
    # Constants between experiments
    frequency = 250  # in Hz
    run_time = 10  # in seconds

    # change policy in run and run_real, zero_policy and flip_and_hold_policy are good to validate simulation



    nsteps = int(run_time * frequency)
    i_steps = 1
    plt.rcParams["figure.figsize"] = (20, 20)  # make graphs BIG


    # # optimize
    # from scipy.optimize import brute
    # # rranges = (slice(0.10, 0.121, 0.01),
    # #            slice(0.0845, 0.0856, 0.0005),
    # #            slice(0.000275, 0.000276, 0.000005),
    # #            slice(0.039, 0.0411, 0.001),
    # #            slice(0.1288, 0.12921, 0.0002),
    # #            slice(0.0000495, 0.0000506, 0.0000005))
    # #
    # # # input tuning
    # rranges = (slice(8, 9.1, 0.2),
    #            slice(0.035, 0.048, 0.001),
    #            slice(0.035, 0.048, 0.001))
    #
    # x0, fval, grid, jout = brute(run, rranges, finish=None, disp=True, full_output=True)
    # print(x0)
    # print(fval)

    # # evaluate
    run_real()
    run(visualize=True)
