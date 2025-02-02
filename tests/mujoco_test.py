#!/usr/bin/env python
# demonstration of markers (visual-only geoms)

import os
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer

MODEL_XML = """
<?xml version="1.0" encoding="utf-8"?>
<mujoco model="qube">
    <compiler angle="radian" meshdir="../meshes/" />
    <option timestep="0.01" integrator="RK4" />
    <size njmax="500" nconmax="100" />

    <visual>
        <global offwidth="2560" offheight="1329" />
    </visual>

    <worldbody>
        <light pos="0.051036 -0.0321818 0.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.9 0.9 0.9" />
        <body name="arm" pos="0 -0.019632 0">
            <inertial pos="0 -0.0425 0" quat="0.707107 0.707107 0 0" mass="0.095" diaginertia="5.7439e-05 5.7439e-05 4.82153e-07" />
            <joint name="base_motor" pos="0 0 0" axis="0 0 1" range="-90 90" damping="0.0005" />
            <joint name="arm_pole" pos="0 0 0" axis="0 1 0" range="-90 90" damping="3e-5" />
            <geom name="qube0:arm" size="0.003186 0.0425" pos="0 -0.0425 0" quat="0.707107 0.707107 0 0" type="cylinder" friction="0 0 0" />
            <body name="pole" pos="0 -0.080312 0">
                <inertial pos="0 0 0.05654" quat="0 1 0 0" mass="0.024" diaginertia="3.34191e-05 3.34191e-05 2.74181e-07" />
                <geom name="qube0:pole" size="0.00478 0.0645" pos="0 0 0.05654" quat="0 1 0 0" type="cylinder" friction="0 0 0" />
            </body>
        </body>
    </worldbody>

    <actuator>
        <general name="motor_rotation" joint="base_motor" ctrllimited="true" ctrlrange="-18 18" gear="0.005 0 0 0 0 0" />
    </actuator>
</mujoco>
"""


def angle_normalize(x: float) -> float:
    return (x % (2 * np.pi)) - np.pi

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
step = 0
ctrl = 1
while True:
    sim.data.ctrl[:] = ctrl
    sim.step()
    ctrl = 0

    theta_before, alpha_before = sim.data.qpos
    theta_dot, alpha_dot = sim.data.qvel

    theta = -1 * angle_normalize(theta_before + np.pi)
    alpha = angle_normalize(alpha_before + np.pi)
    viewer.add_marker(pos=np.array([-0.5, 0, 0.1]), label=f"Theta: {theta} \nAlpha: {alpha}")

    viewer.render()

    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break
