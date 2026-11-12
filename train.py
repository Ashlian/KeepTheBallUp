"""Train the arm to juggle with PPO (stable-baselines3).

Install first:  python3 -m pip install gymnasium stable-baselines3
Train:          python3 train.py              (no viewer, runs as fast as possible)
Keep training:  python3 train.py --continue   (picks up from the saved ppo_juggle.zip)
Watch result:   mjpython train.py --watch     (macOS needs mjpython to open the viewer)
"""
import importlib.util
import os
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.utils import configure_logger

from juggle_env import JuggleEnv

MODEL_FILE = "ppo_juggle"   # saved as ppo_juggle.zip
TOTAL_TIMESTEPS = 2_000_000

# Training graphs only if tensorboard is installed (python3 -m pip install tensorboard)
TB_LOG = "runs/" if importlib.util.find_spec("tensorboard") else None

# Rows left out of the progress table printed during training.
# Only hidden from the terminal: tensorboard (if installed) still records them.
HIDDEN_LOG_KEYS = {
    "ep_rew_mean", "fps", "entropy_loss", "explained_variance",
    "clip_fraction", "clip_range", "approx_kl",
}


class FilteredTerminalOutput(HumanOutputFormat):
    """SB3's printed table, minus the rows in HIDDEN_LOG_KEYS."""

    def write(self, key_values, key_excluded, step=0):
        # Keys look like "rollout/ep_rew_mean"; match on the part after the slash
        kept = {k: v for k, v in key_values.items() if k.split("/")[-1] not in HIDDEN_LOG_KEYS}
        excluded = {k: key_excluded[k] for k in kept if k in key_excluded}
        super().write(kept, excluded, step)


def use_filtered_logger(model, resume):
    """Same logger SB3 would make on its own, with the terminal output filtered."""
    logger = configure_logger(verbose=1, tensorboard_log=TB_LOG, tb_log_name="PPO",
                              reset_num_timesteps=not resume)
    logger.output_formats = [
        FilteredTerminalOutput(sys.stdout) if isinstance(f, HumanOutputFormat) else f
        for f in logger.output_formats
    ]
    model.set_logger(logger)


def train(resume=False):
    env = make_vec_env(JuggleEnv, n_envs=8)  # 8 sims in parallel collect data faster
    if resume and os.path.exists(MODEL_FILE + ".zip"):
        model = PPO.load(MODEL_FILE, env=env, tensorboard_log=TB_LOG)
        print(f"Continuing from {MODEL_FILE}.zip")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=1024,         # steps per env before each update
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            verbose=1,
            tensorboard_log=TB_LOG,
        )
    use_filtered_logger(model, resume)
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=not resume)
    except KeyboardInterrupt:
        print("Stopped early, saving what was learned so far.")
    model.save(MODEL_FILE)
    print(f"Saved {MODEL_FILE}.zip")


def watch():
    env = JuggleEnv(render_mode="human")
    model = PPO.load(MODEL_FILE)
    obs, _ = env.reset()
    env.render()  # opens the viewer
    while env.viewer.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        time.sleep(1 / env.metadata["render_fps"])
    env.close()


if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    else:
        train(resume="--continue" in sys.argv)
