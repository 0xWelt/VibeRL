import torch
import torch.nn as nn

from .base_network import BaseNetwork


class VNetwork(BaseNetwork):
    """Value network for PPO and other policy gradient methods (returns single scalar value for state)."""

    def __init__(self, state_size: int, hidden_size: int = 128, num_hidden_layers: int = 2):
        super().__init__(state_size, hidden_size, num_hidden_layers)

        # Single output for state value
        self.value_head = nn.Linear(hidden_size, 1)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get state value.

        Args:
            x: Input state tensor of shape (batch_size, state_size)

        Returns:
            State value tensor of shape (batch_size,) representing the
            estimated value of the given state
        """
        features = self.forward_backbone(x)
        return self.value_head(features).squeeze(-1)  # Remove last dim


class QNetwork(BaseNetwork):
    """Q-network for value-based methods like DQN (returns Q-values for all actions)."""

    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 128, num_hidden_layers: int = 2
    ):
        super().__init__(state_size, hidden_size, num_hidden_layers)
        self.action_size = action_size

        # Q-value head for all actions
        self.q_head = nn.Linear(hidden_size, action_size)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-values for all actions.

        Args:
            x: Input state tensor of shape (batch_size, state_size)

        Returns:
            Q-values tensor of shape (batch_size, action_size) containing
            Q-values for each action in the given state
        """
        features = self.forward_backbone(x)
        return self.q_head(features)

    def get_q_values(self, state: list | tuple | torch.Tensor) -> torch.Tensor:
        """Get Q-values for a given state.

        Convenience method that handles various input types and ensures
        proper tensor formatting before forward pass.

        Args:
            state: Current state as list, tuple, or tensor

        Returns:
            Q-values tensor of shape (1, action_size) if single state,
            or (batch_size, action_size) if batch of states
        """
        if isinstance(state, list | tuple):
            state = torch.FloatTensor(state)
        else:
            state = torch.FloatTensor(state)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        return self.forward(state)

    def get_action(self, state: list | tuple | torch.Tensor, epsilon: float = 0.0) -> int:
        """Get action using epsilon-greedy policy.

        Implements the epsilon-greedy action selection strategy where:
        - With probability epsilon: choose random action (exploration)
        - With probability 1-epsilon: choose best action (exploitation)

        Args:
            state: Current state as list, tuple, or tensor
            epsilon: Probability of choosing random action (0.0 to 1.0)

        Returns:
            Selected action as integer
        """
        q_values = self.get_q_values(state)

        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_size, (1,)).item()
        else:
            return q_values.argmax(dim=1).item()
