"""Entry point for training and evaluating the PacMan DQN agent."""

import argparse
from collections import deque
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing


class SimpleFrameStack(gym.Wrapper):
    """Fallback frame stack when gymnasium.FrameStack is unavailable."""

    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        h, w = env.observation_space.shape[:2]
        # Grayscale frames produce (H, W); stack along channel axis
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, k),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(self._process_obs(obs))
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._process_obs(obs))
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)

    def _process_obs(self, obs):
        # Ensure obs has channel dimension before stacking
        if obs.ndim == 2:
            obs = obs[:, :, None]
        return obs

from train import DQNTrainer
from evaluate import run_evaluation


def make_env(render_mode: Optional[str] = None):
    """Create Atari MsPacman env with preprocessing and frame stacking."""
    env = gym.make("ALE/MsPacman-v5", render_mode=render_mode, frameskip=1)
    # Resize to 84x84, grayscale, scale rewards disabled, terminal_on_life_loss False
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4)
    env = SimpleFrameStack(env, k=4)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--save-path", type=str, default="pacman_dqn.pt")
    parser.add_argument("--render", action="store_true", help="Render env in real time during training/eval")
    parser.add_argument("--render-final", action="store_true", help="Render only the final training episode")
    args = parser.parse_args()

    render_mode = "human" if (args.render or args.render_final) else None
    env = make_env(render_mode=render_mode)
    num_actions = env.action_space.n

    trainer = DQNTrainer(num_actions=num_actions)

    print(f"Training for {args.train_episodes} episodes...")
    trainer.train(
        env,
        num_episodes=args.train_episodes,
        render=args.render,
        render_final=args.render_final,
    )

    print(f"Saving model to {args.save_path}")
    trainer.save_model(args.save_path)

    if args.eval_episodes > 0:
        print(f"Evaluating for {args.eval_episodes} episodes...")
        avg_reward = run_evaluation(env, trainer, episodes=args.eval_episodes, render=args.render)
        print(f"Average reward over {args.eval_episodes} episodes: {avg_reward:.2f}")
    else:
        print("Skipping evaluation (eval-episodes=0)")


if __name__ == "__main__":
    main()