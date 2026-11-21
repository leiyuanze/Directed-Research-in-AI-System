#!/usr/bin/env python3
"""Train the Module 7 PPO implementation on CartPole without a render window."""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CODE_DIR = Path(__file__).resolve().parent / "code"
sys.path.insert(0, str(CODE_DIR))

import rl_utils  # noqa: E402


def load_ppo_module():
    """Import the starter classes even when its original dependency is gym."""
    path = CODE_DIR / "ppo.py"
    try:
        import gym  # noqa: F401
    except ImportError:
        import gymnasium

        sys.modules["gym"] = gymnasium

    import ppo

    return ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import gym
    except ImportError:
        import gymnasium as gym

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    available_device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("CartPole-v0")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    ppo_module = load_ppo_module()
    agent = ppo_module.PPO(
        state_dim,
        args.hidden_dim,
        action_dim,
        args.actor_lr,
        args.critic_lr,
        args.lmbda,
        args.epochs,
        args.eps,
        args.gamma,
        torch.device(available_device),
    )

    returns, _ = rl_utils.train_on_policy_agent(env, agent, args.episodes)
    env.close()

    episode_returns = [float(value) for value in returns]
    window = 9
    moving_average = rl_utils.moving_average(episode_returns, window).tolist()
    result = {
        "environment": "CartPole-v0",
        "seed": args.seed,
        "device": available_device,
        "episodes": args.episodes,
        "episode_returns": episode_returns,
        "moving_average_window": window,
        "moving_average": moving_average,
        "moving_average_episodes": (
            5 + np.arange(len(moving_average))
        ).tolist(),
        "summary": {
            "first_100_mean": float(np.mean(episode_returns[:100])),
            "last_100_mean": float(np.mean(episode_returns[-100:])),
            "last_100_min": float(np.min(episode_returns[-100:])),
            "last_100_reached_200": int(
                np.sum(np.asarray(episode_returns[-100:]) == 200)
            ),
            "max_return": float(np.max(episode_returns)),
        },
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "gamma": args.gamma,
            "lmbda": args.lmbda,
            "epochs": args.epochs,
            "eps": args.eps,
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "gym": getattr(gym, "__version__", "unknown"),
            "numpy": np.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
