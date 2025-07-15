r"""PPO: Proximal Policy Optimization for stable policy gradient updates.

**Algorithm Overview:**

PPO is a policy gradient method that prevents large policy updates through a
clipped surrogate objective, making training more stable and reliable while
maintaining sample efficiency.

**Key Concepts:**

- **Clipped Surrogate Objective**: Prevents destructive policy updates
- **Generalized Advantage Estimation (GAE)**: Computes stable advantage estimates
- **Multiple PPO Epochs**: Reuses collected data efficiently
- **Policy Network**: $\pi_\theta(a|s)$ for action selection
- **Value Network**: $V_\phi(s)$ for baseline estimation

**Mathematical Foundation:**

**Optimization Objective:**

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

**Advantage Function:**

$$A_t = \delta_t + (\gamma\lambda) \delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \dots$$

**Reference:**
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. Proximal Policy
Optimization Algorithms. *arXiv preprint arXiv:1707.06347* (2017).
[PDF](https://arxiv.org/abs/1707.06347)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from viberl.agents.base import Agent
from viberl.networks.policy_network import PolicyNetwork
from viberl.networks.value_network import VNetwork
from viberl.typing import Action, Trajectory


class PPOAgent(Agent):
    """PPO agent implementation with clipped surrogate objective and GAE.

    This agent implements Proximal Policy Optimization using a clipped surrogate
    objective to prevent large policy updates, along with Generalized Advantage
    Estimation for stable advantage computation.

    Args:
        state_size: Dimension of the state space. Must be positive.
        action_size: Number of possible actions. Must be positive.
        learning_rate: Learning rate for the Adam optimizer. Must be positive.
        gamma: Discount factor for future rewards. Should be in (0, 1].
        lam: GAE lambda parameter for advantage computation. Should be in [0, 1].
        clip_epsilon: PPO clipping parameter. Should be positive.
        value_loss_coef: Coefficient for value loss. Should be positive.
        entropy_coef: Coefficient for entropy bonus. Should be positive.
        max_grad_norm: Maximum gradient norm for clipping. Should be positive.
        ppo_epochs: Number of PPO epochs per update. Must be positive.
        batch_size: Batch size for training. Must be positive.
        hidden_size: Number of neurons in each hidden layer. Must be positive.
        num_hidden_layers: Number of hidden layers. Must be non-negative.
        device: Device for computation ('auto', 'cpu', or 'cuda').

    Raises:
        ValueError: If any parameter is invalid.
    """

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

    def act(self, state: np.ndarray, training: bool = True) -> Action:
        r"""Select action using policy $\pi(a|s;\theta)$.

        Args:
            state: Current state observation.
            training: Whether in training mode (affects exploration).

        Returns:
            Action containing the selected action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.policy_network(state_tensor)
            dist = Categorical(action_probs)

            if training:
                # Training mode: sample from policy distribution
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return Action(action=action.item(), logprobs=log_prob)
            else:
                # Evaluation mode: select most likely action (greedy)
                action = action_probs.argmax().item()
                return Action(action=action)

    def learn(self, trajectory: Trajectory) -> dict[str, float]:
        """Update policy and value networks using PPO clipped objective.

        Args:
            trajectory: Complete trajectory containing transitions with log probabilities.

        Returns:
            Dictionary containing policy loss, value loss, and total loss.
        """
        if not trajectory.transitions:
            return {}

        # Extract data from trajectory
        states = [t.state for t in trajectory.transitions]
        actions = [t.action.action for t in trajectory.transitions]
        rewards = [t.reward for t in trajectory.transitions]
        log_probs = [
            t.action.logprobs.item() if t.action.logprobs is not None else 0.0
            for t in trajectory.transitions
        ]

        # Compute values for each state
        values = []
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                value = self.value_network(state_tensor).squeeze(-1).item()
                values.append(value)

        dones = [t.done for t in trajectory.transitions]

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
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards from episode.
            values: List of state values from value network.
            dones: List of episode termination flags.

        Returns:
            Tuple of (advantages, returns) lists.
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

    def save(self, filepath: str) -> None:
        """Save the agent's neural network parameters to a file.

        Args:
            filepath: Path where to save the model
        """
        torch.save(
            {
                'policy_network': self.policy_network.state_dict(),
                'value_network': self.value_network.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load the agent's neural network parameters from a file.

        Args:
            filepath: Path from which to load the model
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
