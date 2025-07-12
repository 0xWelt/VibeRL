import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from viberl.agents.base import Agent
from viberl.networks.policy_network import PolicyNetwork


class REINFORCEAgent(Agent):
    """REINFORCE policy gradient agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
    ):
        super().__init__(state_size, action_size)
        self.gamma = gamma
        self.policy_network = PolicyNetwork(state_size, action_size, hidden_size, num_hidden_layers)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using current policy."""
        return self.policy_network.act(state)

    def learn(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        **kwargs,
    ) -> dict[str, float]:
        """Perform one learning step using REINFORCE algorithm.

        Args:
            states: List of states from rollout
            actions: List of actions from rollout
            rewards: List of rewards from rollout
        """
        if not rewards:
            return {}

        # Compute returns
        returns = self._compute_returns(rewards)

        # Normalize returns for stability
        returns = torch.FloatTensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Convert states and actions to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)

        # Get action probabilities
        action_probs = self.policy_network(states_tensor)

        # Compute loss
        m = Categorical(action_probs)
        log_probs = m.log_prob(actions_tensor)
        loss = -torch.mean(log_probs * returns)

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'reinforce/policy_loss': loss.item(),
            'reinforce/return_mean': returns.mean().item(),
        }

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """Compute discounted returns."""
        returns = []
        discounted_return = 0
        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        return returns
