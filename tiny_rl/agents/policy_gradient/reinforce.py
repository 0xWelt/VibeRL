from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Neural network policy for REINFORCE algorithm."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

    def act(self, state: np.ndarray) -> int:
        """Select action based on current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)

        # Sample action from probability distribution
        m = Categorical(action_probs)
        action = m.sample()

        return action.item()


class REINFORCEAgent:
    """REINFORCE policy gradient agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_size: int = 128,
    ):
        self.gamma = gamma
        self.policy_network = PolicyNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Storage for episode data
        self.states = []
        self.actions = []
        self.rewards = []

    def select_action(self, state: np.ndarray) -> int:
        """Select action using current policy."""
        return self.policy_network.act(state)

    def store_transition(self, state: np.ndarray, action: int, reward: float):
        """Store transition for training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        """Clear stored transitions."""
        self.states = []
        self.actions = []
        self.rewards = []

    def compute_returns(self, rewards: list[float]) -> list[float]:
        """Compute discounted returns."""
        returns = []
        discounted_return = 0
        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        return returns

    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        if len(self.rewards) == 0:
            return

        # Compute returns
        returns = self.compute_returns(self.rewards)

        # Normalize returns for stability
        returns = torch.FloatTensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Convert states and actions to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)

        # Get action probabilities
        action_probs = self.policy_network(states)

        # Compute loss
        m = Categorical(action_probs)
        log_probs = m.log_prob(actions)
        loss = -torch.mean(log_probs * returns)

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory
        self.clear_memory()

    def save_policy(self, filepath: str):
        """Save policy network."""
        torch.save(self.policy_network.state_dict(), filepath)

    def load_policy(self, filepath: str):
        """Load policy network."""
        self.policy_network.load_state_dict(torch.load(filepath))


def train_reinforce(
    env: gym.Env,
    agent: REINFORCEAgent,
    num_episodes: int = 1000,
    max_steps: int = 1000,
    render_interval: int | None = None,
    save_interval: int | None = None,
    save_path: str | None = None,
    verbose: bool = True,
) -> list[float]:
    """Train REINFORCE agent."""
    scores = []
    recent_scores = deque(maxlen=100)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten grid to vector

        episode_reward = 0

        for _step in range(max_steps):
            # Select action
            action = agent.select_action(state)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward)

            episode_reward += reward
            state = next_state

            # Render if specified
            if render_interval and episode % render_interval == 0:
                env.render()

            if done:
                break

        # Update policy at end of episode
        agent.update_policy()

        scores.append(episode_reward)
        recent_scores.append(episode_reward)

        # Print progress
        if verbose and (episode + 1) % 10 == 0:
            avg_score = np.mean(recent_scores)
            print(
                f'Episode {episode + 1}/{num_episodes}, Average Score (last 100): {avg_score:.2f}'
            )

        # Save model if specified
        if save_interval and save_path and (episode + 1) % save_interval == 0:
            agent.save_policy(f'{save_path}_episode_{episode + 1}.pth')

    return scores


def evaluate_agent(
    env: gym.Env,
    agent: REINFORCEAgent,
    num_episodes: int = 10,
    render: bool = False,
    max_steps: int = 1000,
) -> list[float]:
    """Evaluate trained agent."""
    scores = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()

        episode_reward = 0

        for _step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()
            done = terminated or truncated

            episode_reward += reward
            state = next_state

            if render:
                env.render()

            if done:
                break

        scores.append(episode_reward)
        print(f'Evaluation Episode {episode + 1}: Score = {episode_reward}')

    print(f'Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}')
    return scores
