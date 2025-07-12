import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from viberl.agents.base import Agent
from viberl.networks.value_network import QNetwork


class DQNAgent(Agent):
    """Deep Q-Network (DQN) agent."""

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

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Loss tracking
        self.last_loss = 0.0
        self.steps = 0

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))

        with torch.no_grad():
            q_values = self.q_network.get_q_values(state)
            return q_values.argmax(dim=1).item()

    def learn(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        next_states: list[np.ndarray],
        dones: list[bool],
        **kwargs,
    ) -> dict[str, float]:
        """Perform one learning step using a batch from replay memory.

        Args:
            states: List of states from rollout
            actions: List of actions from rollout
            rewards: List of rewards from rollout
            next_states: List of next states from rollout
            dones: List of done flags from rollout
        """
        self.steps += 1
        metrics = {}

        # Store transitions in memory
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return metrics

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)

        # Convert to tensors
        batch_states = torch.FloatTensor(np.array(batch_states))
        batch_actions = torch.LongTensor(batch_actions)
        batch_rewards = torch.FloatTensor(batch_rewards)
        batch_next_states = torch.FloatTensor(np.array(batch_next_states))
        batch_dones = torch.BoolTensor(batch_dones)

        # Current Q values
        current_q_values = self.q_network(batch_states).gather(1, batch_actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(batch_next_states).max(1)[0]
            target_q_values = batch_rewards + (self.gamma * next_q_values * ~batch_dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track loss
        self.last_loss = loss.item()
        metrics = {'dqn/loss': loss.item(), 'dqn/epsilon': self.epsilon}

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # Update target network
        if self.steps % self.target_update == 0:
            self._update_target_network()

        return metrics

    def _update_target_network(self) -> None:
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
