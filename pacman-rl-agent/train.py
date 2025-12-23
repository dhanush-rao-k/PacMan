import torch
import torch.optim as optim
import torch.nn.functional as F
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer
import numpy as np

class DQNTrainer:
    def __init__(self, num_actions, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995, target_update_freq=1000):
        """
        Initialize DQN trainer
        
        Args:
            num_actions: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate per episode
            target_update_freq: Update target network every N steps
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.q_network = DQNAgent(num_actions).to(self.device)
        self.target_network = DQNAgent(num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.steps = 0

    def _to_tensor_states(self, states):
        """Convert states to NCHW float tensor normalized to [0,1]."""
        tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        if tensor.dim() == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor / 255.0
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = self._to_tensor_states([state])
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
    
    def compute_loss(self, batch_size):
        """Compute TD loss from batch samples"""
        if len(self.replay_buffer) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = self._to_tensor_states(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = self._to_tensor_states(next_states)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s,a)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q(s',a')
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q)
        return loss
    
    def train_episode(self, env, batch_size=32, render=False):
        """Train for one episode."""
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Choose action using epsilon-greedy
            action = self.select_action(state)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if render:
                env.render()
            
            # Store experience in replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            episode_reward += reward
            self.steps += 1
            
            # Sample batch and compute loss
            loss = self.compute_loss(batch_size)
            
            if loss is not None:
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
                self.optimizer.step()
            
            # Update target network every N steps
            if self.steps % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            state = next_state
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return episode_reward
    
    def train(self, env, num_episodes, batch_size=32, eval_freq=100, render=False, render_final=False):
        """Train the DQN agent.

        Args:
            env: The Gymnasium environment.
            num_episodes: Number of training episodes.
            batch_size: Batch size for replay samples.
            eval_freq: Print running average every this many episodes.
            render: Render every episode (slow).
            render_final: Render only the final episode.
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            render_now = render or (render_final and episode == num_episodes - 1)
            reward = self.train_episode(env, batch_size, render=render_now)
            episode_rewards.append(reward)
            
            if (episode + 1) % eval_freq == 0:
                avg_reward = np.mean(episode_rewards[-eval_freq:])
                print(f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {self.epsilon:.3f}")
        
        return episode_rewards
    
    def save_model(self, path):
        """Save trained model"""
        torch.save(self.q_network.state_dict(), path)
    
    def load_model(self, path):
        """Load trained model"""
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
