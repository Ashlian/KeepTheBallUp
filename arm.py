import numpy as np
import mujoco


class UR5eArm:
    """Organized access to the UR5e's joints, actuators, and end effector.

    Knows nothing about balloons, rewards, or RL, so it can be reused anywhere.
    """

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    ACTUATOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    EE_SITE = "attachment_site"  # tip of the wrist, where a paddle would go

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.n_joints = len(self.JOINT_NAMES)

        # Where each joint lives inside qpos / qvel (don't assume they start at 0)
        self.qpos_idx = np.array([model.joint(n).qposadr[0] for n in self.JOINT_NAMES])
        self.qvel_idx = np.array([model.joint(n).dofadr[0] for n in self.JOINT_NAMES])
        self.act_idx = np.array([model.actuator(n).id for n in self.ACTUATOR_NAMES])

        self.ctrl_low = model.actuator_ctrlrange[self.act_idx, 0].copy()
        self.ctrl_high = model.actuator_ctrlrange[self.act_idx, 1].copy()
        self.home_qpos = model.key("home").qpos[self.qpos_idx].copy()

        self.ee_site_id = model.site(self.EE_SITE).id
        self.wrist_body_id = model.body("wrist_1_link").id  # first wrist joint; moves wrist_2/3 and the hand

        # Every geom attached to the arm (used to detect balloon hits)
        base_id = model.body("base").id
        self.geom_ids = {
            g for g in range(model.ngeom)
            if model.body_rootid[model.geom_bodyid[g]] == base_id
        }

    # ---------- reading state ----------
    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.qvel_idx].copy()

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    @property
    def wrist_pos(self) -> np.ndarray:
        """Position of the wrist_1 joint (the body origin sits exactly on the joint)."""
        return self.data.xpos[self.wrist_body_id].copy()

    @property
    def ee_vel(self) -> np.ndarray:
        """Linear velocity of the end effector in world coordinates."""
        vel = np.zeros(6)  # [angular(3), linear(3)]
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                 self.ee_site_id, vel, 0)
        return vel[3:]

    # ---------- commanding ----------
    def set_targets(self, joint_targets: np.ndarray):
        """Command absolute joint angles (radians). Actuators are position servos."""
        self.data.ctrl[self.act_idx] = np.clip(joint_targets, self.ctrl_low, self.ctrl_high)

    def set_targets_from_action(self, action: np.ndarray, low: np.ndarray, high: np.ndarray):
        """Map a policy action in [-1, 1] to joint targets around the home pose.

        low / high are how far each joint may go below / above home (low <= 0 <= high).
        action 0 -> home, +1 -> home + high, -1 -> home + low. The two sides can differ,
        so a joint can swing far one way but only a little the other.

        TODO: try other mappings, e.g. small deltas from the current targets.
        """
        action = np.clip(action, -1.0, 1.0)
        offset = np.where(action >= 0, action * high, -action * low)
        self.set_targets(self.home_qpos + offset)

    def action_from_targets(self, targets: np.ndarray, low: np.ndarray, high: np.ndarray):
        """Inverse of set_targets_from_action: joint targets -> action in [-1, 1]."""
        offset = targets - self.home_qpos
        with np.errstate(divide="ignore", invalid="ignore"):
            action = np.where(offset >= 0, offset / high, -offset / low)
        return np.clip(np.nan_to_num(action), -1.0, 1.0)

    def reset_to_home(self, noise: float = 0.0, rng: np.random.Generator | None = None):
        """Put the joints (and their targets) at the home pose, optionally jittered."""
        q = self.home_qpos.copy()
        if noise > 0 and rng is not None:
            q += rng.uniform(-noise, noise, size=self.n_joints)
        self.data.qpos[self.qpos_idx] = q
        self.data.qvel[self.qvel_idx] = 0.0
        self.set_targets(q)
