import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from viberl.agents.base import Agent
from viberl.networks.value_network import QNetwork
from viberl.typing import Action, Trajectory


class DQNAgent(Agent):
    """DQN: Deep Q-Network combining Q-learning with deep neural networks for human-level control.

    **Key Concepts:**
    • Learns optimal action-value function Q*(s,a) using neural networks
    • Approximates Q-values in high-dimensional state spaces
    • Experience replay buffer removes correlated samples
    • Separate target network Q_θ⁻ for stable target computation
    • Epsilon-greedy exploration balances learning and exploitation

    **Optimization Objective:**
    $$L(\theta) = \\mathbb{E}_{(s,a,r,s') \\sim D}\\left[\\left(r + \\gamma \\max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right]$$

    **Reference:**
    Mnih, V., Kavukcuoglu, K., Silver, D., et al. Human-level control through deep reinforcement learning. *Nature* **518**, 529-533 (2015). [PDF](https://www.nature.com/articles/nature14236)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
    ):
        super().__init__(state_size, action_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon = epsilon_start

        # Neural networks
        self.q_network = QNetwork(state_size, action_size, hidden_size, num_hidden_layers)
        self.target_network = QNetwork(state_size, action_size, hidden_size, num_hidden_layers)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Copy weights to target network
        self._update_target_network()

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

    def act(self, state: np.ndarray, training: bool = True) -> Action:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state observation.
            training: Whether in training mode (affects exploration).

        Returns:
            Action containing the selected action.
        """
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network.get_q_values(state)
                action = q_values.argmax().item()

        return Action(action=action)

    def learn(self, trajectory: Trajectory, **kwargs) -> dict[str, float]:
        """Update Q-network using Q-learning with experience replay.

        Args:
            trajectory: Complete trajectory containing transitions.

        Returns:
            Dictionary containing loss, epsilon, and memory size.
        """
        if not trajectory.transitions:
            return {}

        # Store transitions in memory
        for transition in trajectory.transitions:
            self.memory.append(transition)

        if len(self.memory) < self.batch_size:
            return {'dqn/memory_size': len(self.memory)}

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)

        # Extract batch data
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.LongTensor([t.action.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.BoolTensor([t.done for t in batch])

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self._update_target_network()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {
            'dqn/loss': loss.item(),
            'dqn/epsilon': self.epsilon,
            'dqn/memory_size': len(self.memory),
        }

    def _update_target_network(self) -> None:
        """Synchronize target network with main Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath: str) -> None:
        """Save the agent's neural network parameters to a file.

        Args:
            filepath: Path where to save the model
        """
        torch.save(
            {
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load the agent's neural network parameters from a file.

        Args:
            filepath: Path from which to load the model
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
