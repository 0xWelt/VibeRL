import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from viberl.agents.base import Agent
from viberl.networks.policy_network import PolicyNetwork
from viberl.typing import Action, Trajectory


class REINFORCEAgent(Agent):
    """REINFORCE: Monte-Carlo policy gradient method using complete episode returns.

    **Key Concepts:**
    • Policy gradient method using likelihood ratio trick
    • Monte-Carlo returns for unbiased gradient estimates
    • High-variance but unbiased gradients
    • Requires complete episodes before learning
    • Policy network $\\pi_\theta(a|s)$ for action selection

    **Optimization Objective:**
    $$\nabla_{\theta} J(\theta) = \\mathbb{E}\\left[\\sum_{t=0}^{T} \\log \\pi_\theta(a_t|s_t) G_t\right]$$
    where $G_t = \\sum_{k=0}^{T-t} \\gamma^k r_{t+k}$ is the Monte-Carlo return.

    **Reference:**
    Williams, R.J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning* **8**, 229-256 (1992). [PDF](https://link.springer.com/article/10.1007/BF00992696)
    """

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

    def act(self, state: np.ndarray, training: bool = True) -> Action:
        """Select action using policy π(a|s;θ).

        Args:
            state: Current state observation.
            training: Whether in training mode (affects exploration).

        Returns:
            Action containing the selected action.
        """
        if training:
            # Training mode: sample from policy distribution
            action = self.policy_network.act(state)
        else:
            # Evaluation mode: select most likely action (greedy)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = self.policy_network(state_tensor)
                action = action_probs.argmax().item()

        return Action(action=action)

    def learn(self, trajectory: Trajectory, **kwargs) -> dict[str, float]:
        """Update policy using REINFORCE gradient.

        Args:
            trajectory: Complete episode trajectory.

        Returns:
            Dictionary containing policy loss and return statistics.
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
        """Compute Monte-Carlo returns.

        Args:
            rewards: List of rewards from episode.

        Returns:
            List of discounted returns for each timestep.
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
