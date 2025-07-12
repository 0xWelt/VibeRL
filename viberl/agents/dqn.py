import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from viberl.networks.value_network import ValueNetwork


class DQNAgent:
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
        self.state_size = state_size
        self.action_size = action_size
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
        self.q_network = ValueNetwork(state_size, action_size, hidden_size, num_hidden_layers)
        self.target_network = ValueNetwork(state_size, action_size, hidden_size, num_hidden_layers)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Copy weights to target network
        self.update_target_network()

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))

        with torch.no_grad():
            q_values = self.q_network.get_q_values(state)
            return q_values.argmax(dim=1).item()

    def store_transition(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def update_policy(self):
        """Update the policy using a batch from replay memory."""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_policy(self, filepath: str):
        """Save the policy network."""
        torch.save(self.q_network.state_dict(), filepath)

    def load_policy(self, filepath: str):
        """Load the policy network."""
        self.q_network.load_state_dict(torch.load(filepath))
        self.target_network.load_state_dict(torch.load(filepath))

    def get_metrics(self) -> dict:
        """Get training metrics."""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
        }
