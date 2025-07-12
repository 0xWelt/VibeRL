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
        """Forward pass to get state value."""
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
        """Forward pass to get Q-values for all actions."""
        features = self.forward_backbone(x)
        return self.q_head(features)

    def get_q_values(self, state: list | tuple | torch.Tensor) -> torch.Tensor:
        """Get Q-values for a given state."""
        if isinstance(state, list | tuple):
            state = torch.FloatTensor(state)
        else:
            state = torch.FloatTensor(state)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        return self.forward(state)

    def get_action(self, state: list | tuple | torch.Tensor, epsilon: float = 0.0) -> int:
        """Get action using epsilon-greedy policy."""
        q_values = self.get_q_values(state)

        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_size, (1,)).item()
        else:
            return q_values.argmax(dim=1).item()
