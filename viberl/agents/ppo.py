"""
Proximal Policy Optimization (PPO) agent implementation.

PPO is a policy gradient method that uses a clipped surrogate objective
to prevent large policy updates, making training more stable.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from viberl.networks.policy_network import PolicyNetwork
from viberl.networks.value_network import VNetwork


class PPOAgent:
    """Proximal Policy Optimization (PPO) agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        device: str = 'auto',
    ):
        """
        Initialize PPO agent.

        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            lam: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            batch_size: Batch size for training
            hidden_size: Hidden layer size
            num_hidden_layers: Number of hidden layers
            device: Device to use for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize networks
        self.policy_network = PolicyNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)

        self.value_network = VNetwork(
            state_size=state_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=learning_rate,
        )

        # Storage for experiences
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def act(self, state: np.ndarray) -> tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            value = self.value_network(state_tensor).squeeze(-1)  # Remove last dimension

            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """Store experience for PPO training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear_experiences(self):
        """Clear stored experiences."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def compute_gae(self, rewards: list, values: list, dones: list) -> tuple[list, list]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        gae = 0

        # Ensure values has one more element than rewards for bootstrap
        if len(values) == len(rewards):
            values = values + [0.0]  # Bootstrap with 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        return advantages, returns

    def update(self) -> dict[str, float]:
        """
        Update policy and value networks using PPO.

        Returns:
            Dictionary of training metrics
        """
        if not self.states:
            return {}

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(self.rewards, self.values, self.dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create dataset
        dataset_size = len(states)
        indices = np.arange(dataset_size)

        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
        }

        # PPO epochs
        for _epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                action_probs = self.policy_network(batch_states)
                values = self.value_network(batch_states).squeeze(-1)

                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Update networks
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters()) + list(self.value_network.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Accumulate metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy_loss'] += entropy_loss.item()
                metrics['total_loss'] += total_loss.item()

        # Average metrics over all batches and epochs
        num_batches = (dataset_size + self.batch_size - 1) // self.batch_size
        for key in metrics:
            metrics[key] /= num_batches * self.ppo_epochs

        # Clear experiences after update
        self.clear_experiences()

        return metrics

    def save_policy(self, filepath: str) -> None:
        """Save policy and value networks."""
        torch.save(
            {
                'policy_network_state_dict': self.policy_network.state_dict(),
                'value_network_state_dict': self.value_network.state_dict(),
                'policy_network_config': {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                },
            },
            filepath,
        )

    def load_policy(self, filepath: str) -> None:
        """Load policy and value networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Get action for evaluation."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            if deterministic:
                return action_probs.argmax().item()
            else:
                dist = Categorical(action_probs)
                return dist.sample().item()
