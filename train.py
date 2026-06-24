"""
PPO training script for Cartpole2D-v0.

Usage:
    python train.py [--timesteps 200000] [--workers 4]

Outputs:
    ./logs/        — TensorBoard event files (one sub-dir per run)
    ppo_cartpole.zip — trained policy
"""

import argparse
import time
import numpy as np

# Import both env modules so gymnasium.register() fires in this process.
import cartpole_env       # noqa: F401  (Cartpole2D-v0)
import cartpole3d_env     # noqa: F401  (Cartpole3D-v0)
from cartpole_env import CartpoleEnv
from cartpole3d_env import Cartpole3DEnv

ENV_CLASSES = {"2d": CartpoleEnv, "3d": Cartpole3DEnv}

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def _make_env_factory(seed: int, env_kind: str):
    """Picklable factory for each worker."""
    def _init():
        import cartpole_env, cartpole3d_env  # register both inside the subprocess
        cls = ENV_CLASSES[env_kind]
        env = cls()
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(timesteps: int, n_workers: int, log_dir: str, model_path: str, env_kind: str):
    env_fns = [_make_env_factory(seed=i, env_kind=env_kind) for i in range(n_workers)]
    vec_cls = SubprocVecEnv if n_workers > 1 else DummyVecEnv
    vec_env = vec_cls(env_fns)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        # Hyperparameters tuned for this task; modest enough to be fast.
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=3e-4,
    )

    print(f"Training PPO for {timesteps:,} steps on {n_workers} workers …")
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s")

    model.save(model_path)
    print(f"Model saved to {model_path}.zip")
    vec_env.close()


def evaluate(model_path: str, n_episodes: int = 5, render: bool = False, env_kind: str = "2d"):
    """Run trained policy and print mean ± std return.

    render=True needs `mjpython` on macOS (passive viewer requirement).
    Default is no rendering so `python3 train.py` runs end-to-end.
    """
    model = PPO.load(model_path)

    returns = []
    render_mode = "human" if render else None
    # mjpython on macOS only permits one passive viewer per process — share one env.
    env = ENV_CLASSES[env_kind](render_mode=render_mode)
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_return = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated
            returns.append(ep_return)
            print(f"  Episode {ep+1}: return = {ep_return:.2f}")
    finally:
        env.close()

    mean_r = np.mean(returns)
    std_r  = np.std(returns)
    print(f"\nEval over {n_episodes} episodes: {mean_r:.2f} ± {std_r:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["2d", "3d"], default="2d",
                        help="Which cartpole: 2d (1 axis) or 3d (2 axes).")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-dir", default="./logs/")
    parser.add_argument("--model-path", default=None,
                        help="Defaults to ppo_cartpole_{env}")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load existing model and evaluate.")
    parser.add_argument("--render", action="store_true",
                        help="Render the eval episodes in MuJoCo viewer "
                             "(requires `mjpython train.py --render` on macOS).")
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = f"ppo_cartpole_{args.env}"

    if not args.eval_only:
        train(args.timesteps, args.workers, args.log_dir, args.model_path, args.env)

    evaluate(args.model_path, render=args.render, env_kind=args.env)


if __name__ == "__main__":
    main()
