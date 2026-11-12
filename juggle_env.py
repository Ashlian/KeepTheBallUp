import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from arm import UR5eArm

MODEL_PATH = "models/universal_robots_ur5e/juggle_scene.xml"

# ---------- config ----------
FRAME_SKIP = 10              # physics steps per policy step: 10 x 0.002 s = 50 Hz control
MAX_EPISODE_STEPS = 1500     # 30 s at 50 Hz
# How far (radians) each joint may move below / above its home angle.
#   pan:     +/- pi  -> full 360 degree turn of the base
#   wrist_1, wrist_2: only -0.15 on the low side; past that the paddle tilts into the forearm
# Checked: 0 arm self-collisions across 30k random poses and all corners of this range.
#                        pan     lift  elbow  wrist_1 wrist_2 wrist_3
ACTION_LOW  = np.array([-np.pi, -1.0, -1.0,  -0.15,  -0.15,  -0.5])
ACTION_HIGH = np.array([ np.pi,  1.0,  1.0,   1.0,    1.0,    0.5])

# Balloon spawn (same as script.py)
SPAWN_MIN_RADIUS, SPAWN_MAX_RADIUS = 0.2, 0.55
SPAWN_MIN_Z, SPAWN_MAX_Z = 1.5, 2.25

#! Make hits only come from paddle
#! Make hits only count when launching ball above a certain threshold
#! Limit tilt on the paddle
#! Punish double bounce gimmick

# ---------- reward dials (TODO: tune) ----------
W_ALIVE = 0.1          # Time: incentivize staying up
W_HIT = 1.0            # # of hits (balloon touching the paddle): incentivize
W_ARM_TOUCH = -0.5     # Balloon touching any other part of the arm: punish (once per touch)
W_RADIAL = -0.1        # Radial distance of balloon from the arm's base: punish
W_HORIZ_VEL = -0.0     # Horizontal velocity of the balloon: punish
W_JERK = -0.0         # Jerk / instability (change in action between steps): punish
W_DROP = -5.0          # One-time penalty when the balloon hits the floor
W_TRACK = 0.2          # Paddle under the balloon: incentivize (max W_TRACK per step, when right under it)
TRACK_SCALE = 0.4      # meters; reward drops to ~37% at this horizontal distance, ~5% at 3x it


class JuggleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.arm = UR5eArm(self.model, self.data)

        self.balloon_joint = self.data.joint("balloon 1")
        self.balloon_geom = self.model.geom("balloon").id
        self.floor_geom = self.model.geom("floor").id
        self.paddle_geom = self.model.geom("paddle").id

        self.render_mode = render_mode
        self.viewer = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.arm.n_joints,), dtype=np.float32)
        obs_dim = len(self._get_obs())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.steps = 0
        self.prev_action = np.zeros(self.arm.n_joints)
        self.was_touching_paddle = False
        self.was_touching_body = False

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.arm.reset_to_home(noise=0.05, rng=self.np_random)
        self._spawn_balloon()
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.prev_action = np.zeros(self.arm.n_joints)
        self.was_touching_paddle = False
        self.was_touching_body = False
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.arm.set_targets_from_action(action, ACTION_LOW, ACTION_HIGH)

        hit = False        # balloon started touching the paddle
        arm_touch = False  # balloon started touching any other part of the arm
        dropped = False
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            touching_paddle, touching_body, touching_floor = self._balloon_contacts()
            # Count only when contact starts, so one long touch counts once
            if touching_paddle and not self.was_touching_paddle:
                hit = True  # TODO: your own hit definition (e.g. minimum launch speed)
            if touching_body and not self.was_touching_body:
                arm_touch = True
            self.was_touching_paddle = touching_paddle
            self.was_touching_body = touching_body
            if touching_floor:
                dropped = True
                break

        self.steps += 1
        reward, reward_info = self._compute_reward(action, hit, arm_touch, dropped)
        terminated = dropped
        truncated = self.steps >= MAX_EPISODE_STEPS
        self.prev_action = action

        if self.render_mode == "human":
            self.render()
        info = {"hit": hit, "arm_touch": arm_touch, "reward_terms": reward_info}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ---------- helpers ----------
    def _spawn_balloon(self):
        theta = self.np_random.uniform(0, 2 * np.pi)
        r = np.sqrt(self.np_random.uniform(SPAWN_MIN_RADIUS ** 2, SPAWN_MAX_RADIUS ** 2))
        z = self.np_random.uniform(SPAWN_MIN_Z, SPAWN_MAX_Z)
        self.balloon_joint.qpos[:3] = [r * np.cos(theta), r * np.sin(theta), z]
        self.balloon_joint.qpos[3:] = [1, 0, 0, 0]
        self.balloon_joint.qvel[:] = 0.0

    def _balloon_contacts(self):
        """Returns (touching_paddle, touching_body, touching_floor) for the balloon this
        physics step. "body" means any part of the arm other than the paddle."""
        touching_paddle = touching_body = touching_floor = False
        for c in self.data.contact[:self.data.ncon]:
            if self.balloon_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.balloon_geom else c.geom1
            if other == self.floor_geom:
                touching_floor = True
            elif other == self.paddle_geom:
                touching_paddle = True
            elif other in self.arm.geom_ids:
                touching_body = True
        return touching_paddle, touching_body, touching_floor

    def _balloon_pos(self):
        return self.balloon_joint.qpos[:3].copy()

    def _balloon_vel(self):
        return self.balloon_joint.qvel[:3].copy()

    def _get_obs(self):
        ee = self.arm.ee_pos
        balloon = self._balloon_pos()
        return np.concatenate([
            self.arm.joint_pos,       # 6
            self.arm.joint_vel,       # 6
            ee,                       # 3
            self.arm.ee_vel,          # 3
            balloon,                  # 3
            self._balloon_vel(),      # 3
            balloon - ee,             # 3  where the balloon is relative to the hand
        ]).astype(np.float32)
        # TODO later: LIDAR / camera observations

    def _compute_reward(self, action, hit, arm_touch, dropped):
        balloon = self._balloon_pos()
        balloon_vel = self._balloon_vel()
        # Horizontal (x, y) distance only: the balloon starts higher than the arm can reach,
        # so the goal is to get the paddle underneath it, not up to it.
        # ee_pos is the center of the flange, which is also the paddle's center.
        paddle_dist = np.linalg.norm(balloon[:2] - self.arm.ee_pos[:2])

        terms = {
            "track": W_TRACK * np.exp(-paddle_dist / TRACK_SCALE),
            "alive": W_ALIVE,
            "hit": W_HIT * float(hit),
            "arm_touch": W_ARM_TOUCH * float(arm_touch),
            "radial": W_RADIAL * np.linalg.norm(balloon[:2]),
            "horiz_vel": W_HORIZ_VEL * np.linalg.norm(balloon_vel[:2]),
            "jerk": W_JERK * float(np.sum((action - self.prev_action) ** 2)),
            "drop": W_DROP * float(dropped),
        }
        return float(sum(terms.values())), terms
