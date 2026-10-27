# # train a quadruped dog to stand up straight, live in the mujoco viewer

# import os
# import sys
# import time
# import threading

# import numpy as np
# import mujoco

# # config
# MODEL_PATH = "go2/scene.xml"
# BODY_NAME = "base"
# POLICY_FILE = "policy.npz"
# CONTROL_DECIMATION = 10
# EPISODE_STEPS = 250
# TARGET_HEIGHT = 0.28
# LOG_EVERY = 5
# DEFAULT_POSE = np.array([0.0, 1.4, -2.5] * 4)
# ACTION_SCALE = np.array([0.8, 1.2, 1.2] * 4)

# # reward dials: change these multipliers and re-run to see the effect
# W_HEIGHT = 1.0      # stand tall
# W_UPRIGHT = 0.5     # stay level, don't tip over
# W_ALIVE = 0.3       # bonus for genuinely standing
# W_EFFORT = -0.001   # penalty for thrashing


# # reward earned each control step
# def compute_reward(obs_parts, action):
#     upright = obs_parts["upright"]
#     height = obs_parts["height"]
#     jnt_vel = obs_parts["jnt_vel"]

#     height_score = 1.0 - min(1.0, abs(TARGET_HEIGHT - height) / TARGET_HEIGHT)
#     upright_score = max(0.0, upright)
#     alive_score = 1.0 if (upright > 0.7 and height > 0.7 * TARGET_HEIGHT) else 0.0
#     effort_score = float(np.sum(jnt_vel ** 2))

#     return (
#         W_HEIGHT * height_score
#         + W_UPRIGHT * upright_score
#         + W_ALIVE * alive_score
#         + W_EFFORT * effort_score
#     )


# # one mujoco sim of the dog plus observe/act helpers
# class DogSim:
#     def __init__(self):
#         self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
#         self.data = mujoco.MjData(self.model)
#         self.start_key = self.model.key("crouch").id
#         self.n_act = self.model.nu
#         self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
#         self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
#         self.jnt_qpos = 7 + np.arange(self.n_act)
#         self.jnt_qvel = 6 + np.arange(self.n_act)
#         self.obs_dim = len(self.make_obs())

#     def reset(self):
#         mujoco.mj_resetDataKeyframe(self.model, self.data, self.start_key)
#         mujoco.mj_forward(self.model, self.data)
#         return self.make_obs()

#     def obs_parts(self):
#         d = self.data
#         R = d.xmat[self.model.body(BODY_NAME).id].reshape(3, 3)
#         return {
#             "upright": float(R[2, 2]),
#             "height": float(d.xpos[self.model.body(BODY_NAME).id][2]),
#             "jnt_pos": d.qpos[self.jnt_qpos].copy(),
#             "jnt_vel": d.qvel[self.jnt_qvel].copy(),
#             "lin_vel": d.qvel[0:3].copy(),
#             "ang_vel": d.qvel[3:6].copy(),
#         }

#     def make_obs(self):
#         p = self.obs_parts()
#         return np.concatenate([
#             [p["upright"], p["height"]],
#             p["jnt_pos"], p["jnt_vel"], p["lin_vel"], p["ang_vel"],
#         ]).astype(np.float64)

#     def act(self, action):
#         action = np.clip(action, -1.0, 1.0)
#         target = np.clip(DEFAULT_POSE + action * ACTION_SCALE, self.ctrl_low, self.ctrl_high)
#         self.data.ctrl[:] = target
#         for _ in range(CONTROL_DECIMATION):
#             mujoco.mj_step(self.model, self.data)


# # running mean/std of observations
# class RunningNorm:
#     def __init__(self, dim):
#         self.n = 1.0
#         self.mean = np.zeros(dim)
#         self.m2 = np.ones(dim)

#     def observe(self, x):
#         self.n += 1
#         delta = x - self.mean
#         self.mean += delta / self.n
#         self.m2 += delta * (x - self.mean)

#     def normalize(self, x):
#         std = np.sqrt(self.m2 / self.n)
#         return (x - self.mean) / (std + 1e-8)


# # affine policy mapping an observation to joint actions in [-1, 1]
# def policy_action(policy, norm, obs):
#     feat = np.append(norm.normalize(obs), 1.0)
#     return np.tanh(policy @ feat)


# # augmented random search settings
# N_DIRECTIONS = 16
# N_BEST = 8
# NOISE_STD = 0.06
# STEP_SIZE = 0.04


# # the learner: jiggle the policy weights, keep the directions that earn more reward
# class Trainer:
#     def __init__(self, shared, load=True):
#         self.sim = DogSim()
#         self.norm = RunningNorm(self.sim.obs_dim)
#         self.policy = np.zeros((self.sim.n_act, self.sim.obs_dim + 1))
#         self.shared = shared
#         self.iteration = 0
#         self.loaded = load and self.load()

#     # load saved weights if the file is there (delete the file to train fresh)
#     def load(self):
#         if not os.path.exists(POLICY_FILE):
#             return False
#         d = np.load(POLICY_FILE)
#         if d["policy"].shape != self.policy.shape:
#             return False
#         self.policy = d["policy"]
#         self.norm.mean, self.norm.m2, self.norm.n = d["mean"], d["m2"], float(d["n"])
#         return True

#     # write the current weights to disk
#     def save(self):
#         np.savez(POLICY_FILE, policy=self.policy,
#                  mean=self.norm.mean, m2=self.norm.m2, n=self.norm.n)

#     def rollout(self, policy, record=True):
#         obs = self.sim.reset()
#         total = 0.0
#         for _ in range(EPISODE_STEPS):
#             if record:
#                 self.norm.observe(obs)
#             action = policy_action(policy, self.norm, obs)
#             self.sim.act(action)
#             obs = self.sim.make_obs()
#             total += compute_reward(self.sim.obs_parts(), action)
#         return total

#     def step(self):
#         deltas = [np.random.randn(*self.policy.shape) for _ in range(N_DIRECTIONS)]
#         rewards = []
#         for d in deltas:
#             rewards.append((self.rollout(self.policy + NOISE_STD * d),
#                             self.rollout(self.policy - NOISE_STD * d)))

#         order = sorted(range(N_DIRECTIONS), key=lambda i: -max(rewards[i]))[:N_BEST]
#         sigma = np.std([r for i in order for r in rewards[i]]) + 1e-6

#         update = np.zeros_like(self.policy)
#         for i in order:
#             r_plus, r_minus = rewards[i]
#             update += (r_plus - r_minus) * deltas[i]
#         self.policy += (STEP_SIZE / (N_BEST * sigma)) * update

#         self.iteration += 1
#         eval_reward = self.rollout(self.policy, record=False)

#         with self.shared["lock"]:
#             self.shared["policy"] = self.policy.copy()
#             self.shared["mean"] = self.norm.mean.copy()
#             self.shared["m2"] = self.norm.m2.copy()
#             self.shared["n"] = self.norm.n
#             self.shared["iteration"] = self.iteration
#             self.shared["reward"] = eval_reward
#         return eval_reward


# # background training loop that prints progress every LOG_EVERY iterations
# def train_loop(shared):
#     trainer = Trainer(shared)
#     print(f"loaded saved policy from {POLICY_FILE}" if trainer.loaded
#           else f"no {POLICY_FILE} found, training from scratch")
#     with shared["lock"]:
#         shared["policy"] = trainer.policy.copy()
#         shared["mean"] = trainer.norm.mean.copy()
#         shared["m2"] = trainer.norm.m2.copy()
#         shared["n"] = trainer.norm.n
#     print(f"{'iter':>6}  {'reward':>8}  {'height':>7}  upright")
#     while not shared["stop"]:
#         r = trainer.step()
#         if trainer.iteration % LOG_EVERY == 0:
#             trainer.save()
#             p = trainer.sim.obs_parts()
#             print(f"{trainer.iteration:6d}  {r:8.1f}  {p['height']:6.3f}m  {p['upright']:5.2f}", flush=True)


# # headless sanity check, no viewer
# def selftest():
#     shared = {"lock": threading.Lock(), "stop": False}
#     trainer = Trainer(shared, load=False)
#     print(f"obs_dim={trainer.sim.obs_dim}  n_act={trainer.sim.n_act}")
#     print(f"random-policy reward: {trainer.rollout(trainer.policy, record=False):.1f}")
#     for _ in range(8):
#         print(f"iter {trainer.iteration + 1:3d}  reward {trainer.step():8.1f}")
#     print("selftest OK")


# # live viewer on the main thread (required on macos) with training in the background
# def run_with_viewer():
#     import mujoco.viewer

#     shared = {
#         "lock": threading.Lock(), "stop": False,
#         "policy": None, "mean": None, "m2": None, "n": 1.0,
#         "iteration": 0, "reward": 0.0,
#     }
#     threading.Thread(target=train_loop, args=(shared,), daemon=True).start()

#     show = DogSim()
#     show.reset()
#     norm = RunningNorm(show.obs_dim)

#     with mujoco.viewer.launch_passive(show.model, show.data) as viewer:
#         last_reset = time.time()
#         while viewer.is_running():
#             with shared["lock"]:
#                 policy = shared["policy"]
#                 if shared["mean"] is not None:
#                     norm.mean = shared["mean"]
#                     norm.m2 = shared["m2"]
#                     norm.n = shared["n"]

#             if policy is not None:
#                 show.act(policy_action(policy, norm, show.make_obs()))

#             if time.time() - last_reset > 6.0 or show.obs_parts()["height"] < 0.08:
#                 show.reset()
#                 last_reset = time.time()

#             viewer.sync()
#             time.sleep(0.025)

#         shared["stop"] = True


# if __name__ == "__main__":
#     if "--selftest" in sys.argv:
#         selftest()
#     else:
#         run_with_viewer()
