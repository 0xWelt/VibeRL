r"""REINFORCE: Monte-Carlo policy gradient method for reinforcement learning.

**Algorithm Overview:**

REINFORCE is a policy gradient method that uses complete episode returns to update
the policy parameters. It directly optimizes the policy without requiring a value
function, making it conceptually simple but potentially high-variance.

**Key Concepts:**

- **Policy Gradient**: Directly optimizes policy parameters $\theta$ to maximize expected return
- **Likelihood Ratio Trick**: Uses $\nabla_\theta \log \pi_\theta(a|s)$ for gradient computation
- **Monte-Carlo Returns**: Uses complete episode returns $G_t$ for unbiased estimates
- **High Variance**: Large variance in gradient estimates but unbiased
- **Episode-based Learning**: Requires complete episodes before parameter updates

**Mathematical Foundation:**

**Optimization Objective:**

$$\nabla_{\theta} J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \log \pi_{\theta}(a_t|s_t) G_t\right]$$

**Return Calculation:**

$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

**Reference:**
Williams, R.J. Simple statistical gradient-following algorithms for connectionist
reinforcement learning. *Machine Learning* **8**, 229-256 (1992).
[PDF](https://link.springer.com/article/10.1007/BF00992696)
"""

import os

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from viberl.agents.base import Agent
from viberl.networks.policy_network import PolicyNetwork
from viberl.typing import Action, Trajectory


class REINFORCEAgent(Agent):
    """REINFORCE agent implementation with policy gradient optimization.

    This agent implements the REINFORCE algorithm using a policy network to directly
    optimize the policy parameters via Monte-Carlo gradient estimates.

    Args:
        state_size: Dimension of the state space. Must be positive.
        action_size: Number of possible actions. Must be positive.
        learning_rate: Learning rate for the Adam optimizer. Must be positive.
        gamma: Discount factor for future rewards. Should be in (0, 1].
        hidden_size: Number of neurons in each hidden layer. Must be positive.
        num_hidden_layers: Number of hidden layers. Must be non-negative.

    Raises:
        ValueError: If any parameter is invalid (e.g., negative dimensions).
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
        """Initialize the REINFORCE agent."""
        super().__init__(state_size, action_size)

        if state_size <= 0:
            raise ValueError(f'state_size must be positive, got {state_size}')
        if action_size <= 0:
            raise ValueError(f'action_size must be positive, got {action_size}')
        if learning_rate <= 0:
            raise ValueError(f'learning_rate must be positive, got {learning_rate}')
        if not 0 < gamma <= 1:
            raise ValueError(f'gamma must be in (0, 1], got {gamma}')
        if hidden_size <= 0:
            raise ValueError(f'hidden_size must be positive, got {hidden_size}')
        if num_hidden_layers < 0:
            raise ValueError(f'num_hidden_layers must be non-negative, got {num_hidden_layers}')

        self.gamma = gamma
        self.policy_network = PolicyNetwork(state_size, action_size, hidden_size, num_hidden_layers)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def act(self, state: np.ndarray, training: bool = True) -> Action:
        """Select action using the current policy.

        Uses the policy network to compute action probabilities and selects
        an action based on the current mode (training vs evaluation).

        In training mode, samples from the policy distribution to ensure
        exploration. In evaluation mode, selects the most probable action
        (greedy selection).

        Args:
            state: Current state observation as a numpy array. Should have shape
                (state_size,) or be convertible to a tensor of that shape.
            training: Whether in training mode. If True, samples from the policy
                distribution for exploration. If False, selects the action with
                highest probability (greedy selection).

        Returns:
            Action containing the selected action index as an integer.

        Raises:
            ValueError: If the state has incorrect shape or type.
            RuntimeError: If there's an error during action computation.
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

    def learn(self, trajectory: Trajectory) -> dict[str, float]:
        """Update policy parameters using the REINFORCE gradient algorithm.

        Computes the policy gradient using the likelihood ratio trick and updates
        the policy network parameters. Uses complete episode returns (Monte-Carlo
        estimates) for gradient computation.

        The method:
        1. Extracts states, actions, and rewards from the trajectory
        2. Computes discounted returns for each timestep
        3. Normalizes returns for training stability
        4. Computes policy loss using log probabilities and returns
        5. Updates policy parameters via backpropagation

        Args:
            trajectory: Complete episode trajectory containing all transitions
                from the episode. Must have at least one transition.

        Returns:
            Dictionary containing training metrics:
            - 'reinforce/policy_loss': Policy loss value (float)
            - 'reinforce/return_mean': Mean of normalized returns (float)

        Raises:
            ValueError: If trajectory is empty or contains invalid data.
            RuntimeError: If there's an error during gradient computation.
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
        r"""Compute discounted Monte-Carlo returns for each timestep.

        Calculates the cumulative discounted return from each timestep to the end
        of the episode. Uses the formula:

        $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$

        This is computed efficiently in reverse order using dynamic programming,
        starting from the final timestep and working backwards.

        Args:
            rewards: List of rewards from the episode, ordered chronologically.
                Each element represents the reward received at a timestep.

        Returns:
            List of discounted returns for each timestep, in the same order
            as the input rewards. The i-th element corresponds to the return
            from timestep i to the end of the episode.

        Example:
            >>> rewards = [1.0, 2.0, 3.0]
            >>> gamma = 0.9
            >>> returns = [1 + 0.9*2 + 0.9^2*3, 2 + 0.9*3, 3]
            >>> # returns = [5.23, 4.7, 3.0]
        """
        returns = []
        discounted_return = 0
        for reward in reversed(rewards):
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        return returns

    def save(self, filepath: str) -> None:
        """Save the agent's policy network parameters to a file.

        Saves the complete state of the policy network, including all parameters
        and buffers. The saved file can be loaded later to restore the exact
        same policy.

        Args:
            filepath: Path where to save the model. Should include the .pth extension.
                The directory will be created if it doesn't exist.

        Raises:
            IOError: If there's an error writing to the file.
            ValueError: If filepath is empty or invalid.

        Example:
            >>> agent.save('models/reinforce_policy.pth')
            >>> # Later: agent.load('models/reinforce_policy.pth')
        """
        if not filepath:
            raise ValueError('filepath cannot be empty')

        # Ensure directory exists
        import os

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        torch.save(self.policy_network.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        """Load the agent's policy network parameters from a file.

        Loads previously saved policy network parameters from disk. Can be used
        to restore an agent to a previously saved state.

        Args:
            filepath: Path from which to load the model. Should point to a .pth file
                created by the save() method.

        Raises:
            IOError: If the file doesn't exist or can't be read.
            ValueError: If filepath is empty or the file contains invalid data.
            RuntimeError: If there's a mismatch between saved and current network architecture.

        Example:
            >>> agent.load('models/reinforce_policy.pth')
            >>> # Agent is now restored to the saved state
        """
        if not filepath:
            raise ValueError('filepath cannot be empty')

        if not os.path.exists(filepath):
            raise OSError(f'File not found: {filepath}')

        state_dict = torch.load(filepath, map_location='cpu')

        # Verify compatibility
        current_params = self.policy_network.state_dict()
        if state_dict.keys() != current_params.keys():
            raise RuntimeError(
                'Network architecture mismatch: saved model has different parameters'
            )

        self.policy_network.load_state_dict(state_dict)
