import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("models/universal_robots_ur5e/juggle_scene.xml")
data = mujoco.MjData(model)
rng = np.random.default_rng()

def randomize_start(model, data):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    # Balloon random starting pos
    max_radius = 0.6
    min_radius = 0.2
    theta = rng.uniform(0, 2*np.pi)
    r = np.sqrt(rng.uniform(min_radius ** 2, max_radius ** 2))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.uniform(1.5, 2.25)
    data.joint("balloon 1").qpos[:3] = [x, y, z]


floor_id = model.geom("floor").id
balloon_id = model.geom("balloon").id

def balloon_hit_floor(data):
    # data.contact lists every pair of geoms touching this step
    for c in data.contact[:data.ncon]:
        if {c.geom1, c.geom2} == {floor_id, balloon_id}:
            return True
    return False


'''
Rewards:

Radial Distance - punish
Horizontal Velocity - punish
Time - incentivize
# of Hits (own definition of hits) - incentivize


'''

'''
Later expansion:

LIDAR Sensor?
Camera?
'''


with mujoco.viewer.launch_passive(model, data) as viewer:
    randomize_start(model, data)
    while viewer.is_running():
        t0 = time.time()
        mujoco.mj_step(model, data)
        if balloon_hit_floor(data):
            randomize_start(model, data)
        viewer.sync()
        # run at roughly real time
        # time.sleep(max(0, model.opt.timestep - (time.time() - t0)))
        time.sleep(max(0, model.opt.timestep - (time.time() - t0)))