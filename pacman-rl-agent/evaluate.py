"""Evaluation helpers for the PacMan DQN agent."""

import torch
import numpy as np


def run_evaluation(env, trainer, episodes=5, render: bool = False):
	"""Run evaluation episodes without training and return average reward."""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	trainer.q_network.eval()

	rewards = []
	for _ in range(episodes):
		state, _ = env.reset()
		done = False
		episode_reward = 0

		while not done:
			with torch.no_grad():
				state_array = np.array(state, dtype=np.float32) / 255.0
				state_tensor = torch.as_tensor(state_array, device=device)
				if state_tensor.dim() == 3:
					state_tensor = state_tensor.permute(2, 0, 1)
				state_tensor = state_tensor.unsqueeze(0)
				q_values = trainer.q_network(state_tensor)
				action = q_values.argmax(dim=1).item()

			next_state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			episode_reward += reward
			state = next_state

			if render:
				env.render()

		rewards.append(episode_reward)

	trainer.q_network.train()
	return float(np.mean(rewards))
