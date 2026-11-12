"""Physics sandbox: watch the juggle scene without training anything.

Run:  mjpython script.py            arm holds its home pose
      mjpython script.py --random   arm makes random moves
      mjpython script.py --manual   you drive the arm with the viewer's Control sliders
                                    (right-hand panel > Control, one slider per joint)

Uses JuggleEnv, so spawning, hit detection, and rewards match training exactly.
Handy after changing juggle_scene.xml (bounciness, mass, drag) or the reward weights.
"""
import sys
import time

import numpy as np

from juggle_env import JuggleEnv, FRAME_SKIP, ACTION_LOW, ACTION_HIGH

MODE = "manual" if "--manual" in sys.argv else "random" if "--random" in sys.argv else "hold"

env = JuggleEnv(render_mode="human")
step_time = env.model.opt.timestep * FRAME_SKIP  # sim seconds per env.step
arm = env.arm


def slider_action():
    """Turn the viewer's Control sliders (joint angle targets) into a policy action.

    Goes through the same action mapping as training, so you can only move each joint
    as far as the policy can (home + ACTION_LOW .. home + ACTION_HIGH).
    Past that, the slider snaps back.
    """
    targets = env.data.ctrl[arm.act_idx]
    return arm.action_from_targets(targets, ACTION_LOW, ACTION_HIGH)


obs, _ = env.reset()
env.render()  # opens the viewer
episode_reward, hitss = 0.0, 0
if MODE == "manual":
    print("Manual mode: open the right-hand panel > Control and drag the joint sliders.")

while env.viewer.is_running():
    t0 = time.time()
    if MODE == "manual":
        action = slider_action()
    elif MODE == "random":
        action = env.action_space.sample()
    else:
        action = np.zeros(arm.n_joints)

    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    hits += info["hit"]
    if MODE == "manual" and info["hit"]:
        print(f"hit! ({hits} this episode)")

    if terminated or truncated:
        print(f"episode over after {env.steps * step_time:.1f}s  reward {episode_reward:.2f}  hits {hits}")
        obs, _ = env.reset()  # also puts the sliders back at home
        episode_reward, hits = 0.0, 0

    time.sleep(max(0, step_time - (time.time() - t0)))  # run at roughly real time

env.close()
