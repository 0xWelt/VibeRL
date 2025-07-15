import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from viberl.agents.base import Agent
from viberl.networks.policy_network import PolicyNetwork
from viberl.typing import Action, Trajectory


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
        """Initialize REINFORCE agent.

        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            learning_rate: Learning rate for policy optimization
            gamma: Discount factor for future rewards (0.0 to 1.0)
            hidden_size: Size of hidden layers in policy network
            num_hidden_layers: Number of hidden layers in policy network
        """
        super().__init__(state_size, action_size)
        self.gamma = gamma
        self.policy_network = PolicyNetwork(state_size, action_size, hidden_size, num_hidden_layers)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def act(self, state: np.ndarray, training: bool = True) -> Action:
        """Select action using current policy.

        Args:
            state: Current state as numpy array
            training: Whether in training mode (affects action selection)

        Returns:
            Action object containing the selected action
        """
        action = self.policy_network.act(state)
        return Action(action=action)

    def learn(
        self,
        trajectory: Trajectory,
        **kwargs,
    ) -> dict[str, float]:
        """Perform one learning step using REINFORCE algorithm.

        Args:
            trajectory: A complete trajectory containing transitions
        """
        if not trajectory.transitions:
            return {}

        # Extract data from trajectory
        states = [t.state for t in trajectory.transitions]
        actions = [t.action.action for t in trajectory.transitions]
        rewards = [t.reward for t in trajectory.transitions]

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
        """Compute discounted returns using Monte Carlo method.

        Calculates the discounted return for each timestep by working backwards
        from the end of the episode. Uses the formula:
        G_t = r_t + gamma * G_{t+1}

        Args:
            rewards: List of rewards from a complete episode

        Returns:
            List of discounted returns for each timestep

        Example:
            >>> rewards = [1, 2, 3]
            >>> returns = [1 + 0.99 * (2 + 0.99 * 3), 2 + 0.99 * 3, 3]
            >>> # returns ≈ [5.9401, 4.97, 3.0]
        """
        returns = []
        discounted_return = 0
        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        return returns

    def save(self, filepath: str) -> None:
        """Save the agent's neural network parameters to a file.

        Args:
            filepath: Path where to save the model
        """
        torch.save(self.policy_network.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """Load the agent's neural network parameters from a file.

        Args:
            filepath: Path from which to load the model
        """
        state_dict = torch.load(filepath, map_location='cpu')
        self.policy_network.load_state_dict(state_dict)
