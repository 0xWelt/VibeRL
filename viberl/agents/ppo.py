"""
Proximal Policy Optimization (PPO) agent implementation.

PPO is a policy gradient method that uses a clipped surrogate objective
to prevent large policy updates, making training more stable.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from viberl.agents.base import Agent
from viberl.networks.policy_network import PolicyNetwork
from viberl.networks.value_network import VNetwork


class PPOAgent(Agent):
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
        super().__init__(state_size, action_size)
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

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()

        return action.item()

    def learn(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        log_probs: list[float],
        values: list[float],
        dones: list[bool],
        **kwargs,
    ) -> dict[str, float]:
        """Update policy and value networks using PPO.

        Args:
            states: List of states from rollout
            actions: List of actions from rollout
            rewards: List of rewards from rollout
            log_probs: List of log probabilities from rollout
            values: List of state values from rollout
            dones: List of done flags from rollout
        """
        if not rewards:
            return {}

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)

        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages (handle small sample sizes)
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )
        else:
            advantages_tensor = advantages_tensor - advantages_tensor.mean()

        # Create dataset
        dataset_size = len(states)
        indices = np.arange(dataset_size)

        metrics = {
            'ppo/policy_loss': 0.0,
            'ppo/value_loss': 0.0,
            'ppo/entropy_loss': 0.0,
            'ppo/total_loss': 0.0,
        }

        # PPO epochs
        for _epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # Forward pass
                action_probs = self.policy_network(batch_states)
                # Ensure action_probs are valid probabilities
                action_probs = torch.clamp(action_probs, 1e-8, 1 - 1e-8)
                action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)

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
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns.squeeze())

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
                metrics['ppo/policy_loss'] += policy_loss.item()
                metrics['ppo/value_loss'] += value_loss.item()
                metrics['ppo/entropy_loss'] += entropy_loss.item()
                metrics['ppo/total_loss'] += total_loss.item()

        # Average metrics over all batches and epochs
        num_batches = (dataset_size + self.batch_size - 1) // self.batch_size
        for key in metrics:
            metrics[key] /= num_batches * self.ppo_epochs

        return metrics

    def _compute_gae(self, rewards: list, values: list, dones: list) -> tuple[list, list]:
        """Compute Generalized Advantage Estimation (GAE)."""
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
