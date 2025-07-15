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
    """Deep Q-Network (DQN) agent with experience replay and target network.

    Implements the DQN algorithm with the following features:
    - Experience replay buffer for stable learning
    - Target network for reduced correlation
    - Epsilon-greedy exploration strategy
    - Batch learning for improved sample efficiency

    The algorithm learns Q-values that approximate the optimal action-value function
    using the Bellman equation: Q(s,a) = r + gamma * max_a' Q(s',a')
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
        """Initialize DQN agent with comprehensive hyperparameter configuration.

        Args:
            state_size: Size of the state space (input features to network)
            action_size: Number of possible actions (output dimension)
            learning_rate: Learning rate for Adam optimizer (typically 1e-4 to 1e-2)
            gamma: Discount factor for future rewards (0.9 to 0.99, default 0.99)
            epsilon_start: Initial exploration rate (1.0 for full exploration)
            epsilon_end: Minimum exploration rate (0.01 for minimal exploration)
            epsilon_decay: Decay rate for epsilon per episode (0.995 for gradual decay)
            memory_size: Size of experience replay buffer (1000 to 100000)
            batch_size: Number of samples per training batch (32 to 256)
            target_update: Frequency of target network updates (in episodes)
            hidden_size: Size of hidden layers in Q-network (64 to 512)
            num_hidden_layers: Number of hidden layers (2 to 4)

        Note:
            The epsilon parameter starts at epsilon_start and decays exponentially:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        """
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

        Implements the epsilon-greedy action selection strategy:
        - In training mode: With probability epsilon, choose random action (exploration)
        - In training mode: With probability 1-epsilon, choose best action (exploitation)
        - In evaluation mode: Always choose best action (greedy)

        Args:
            state: Current state as numpy array
            training: Whether in training mode (affects epsilon-greedy behavior)

        Returns:
            Action object containing the selected action

        Example:
            >>> agent = DQNAgent(state_size=4, action_size=2)
            >>> state = np.array([0.1, 0.2, 0.3, 0.4])
            >>> action = agent.act(state)  # Uses epsilon-greedy
            >>> action = agent.act(state, training=False)  # Greedy action
        """
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network.get_q_values(state)
                action = q_values.argmax().item()

        return Action(action=action)

    def learn(
        self,
        trajectory: Trajectory,
        **kwargs,
    ) -> dict[str, float]:
        """Perform one learning step using Q-learning with experience replay.

        The algorithm follows these steps:
        1. Store new transitions in replay buffer
        2. Sample a batch from memory if enough experiences exist
        3. Compute Q-learning targets using target network
        4. Update Q-network using mean squared error loss
        5. Decay epsilon for exploration
        6. Update target network periodically

        The Q-learning update rule:
        L = (r + gamma * max_a' Q_target(s', a') - Q(s, a))²

        Args:
            trajectory: Complete trajectory containing transitions to learn from

        Returns:
            Dictionary containing training metrics:
            - 'dqn/loss': Mean squared error loss
            - 'dqn/epsilon': Current exploration rate
            - 'dqn/memory_size': Number of experiences in buffer

        Note:
            Learning only occurs when the replay buffer contains at least
            batch_size experiences to ensure stable training.
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
        """Synchronize target network with main Q-network.

        Performs a hard update by copying all parameters from the main Q-network
        to the target Q-network. This helps stabilize learning by providing
        a consistent target for Q-value estimation.

        The target network is updated every `target_update` episodes to balance
        between stable targets and adapting to new policy improvements.

        Note:
            This implementation uses hard updates (full parameter copy) rather than
            soft updates (exponential moving average) for simplicity and stability.
        """
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
