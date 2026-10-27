import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("models/universal_robots_ur5e/juggle_scene.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        # run at roughly real time
        # time.sleep(max(0, model.opt.timestep - (time.time() - t0)))
        time.sleep(max(0, model.opt.timestep - (time.time() - t0)))