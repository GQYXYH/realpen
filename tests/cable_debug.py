import os
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer, load_model_from_path


def angle_normalize(x: float) -> float:
    return (x % (2 * np.pi)) - np.pi

model = load_model_from_path("../gym_brt/data/xml/qube_cable_long.xml")
sim = MjSim(model)
viewer = MjViewer(sim)
step = 0
ctrl = 10
    
print(model.get_xml())

while True:
    sim.data.ctrl[:] = ctrl
    sim.step()          
    ctrl = 10    

    #theta_before, alpha_before = sim.data.qpos
    #theta_dot, alpha_dot = sim.data.qvel

    #theta = -1 * angle_normalize(theta_before + np.pi)
    #alpha = angle_normalize(alpha_before + np.pi)
    #viewer.add_marker(pos=np.array([-0.5, 0, 0.1]), label=f"Theta: {theta} \nAlpha: {alpha}")

    viewer.render()

    step += 1
    if step > 100 and os.getenv('TESTING') is not None:
        break