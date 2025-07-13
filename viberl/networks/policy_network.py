import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base_network import BaseNetwork


class PolicyNetwork(BaseNetwork):
    """Policy network for policy gradient methods like REINFORCE."""

    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 128, num_hidden_layers: int = 2
    ):
        super().__init__(state_size, hidden_size, num_hidden_layers)
        self.action_size = action_size

        # Policy head
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get action probabilities.

        Processes state features through the backbone network and policy head
        to produce normalized action probabilities.

        Args:
            x: Input state tensor of shape (batch_size, state_size)

        Returns:
            Action probabilities tensor of shape (batch_size, action_size)
            with values summing to 1 along the action dimension
        """
        features = self.forward_backbone(x)
        action_logits = self.policy_head(features)
        return self.softmax(action_logits)

    def act(self, state: list | tuple | torch.Tensor, deterministic: bool = False) -> int:
        """Select action based on current policy.

        Args:
            state: Current state as list, tuple, or tensor
            deterministic: If True, always returns the most probable action.
                          If False, samples from the action distribution.

        Returns:
            Selected action as integer
        """
        if isinstance(state, list | tuple):
            state = torch.FloatTensor(state)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)

        action_probs = self.forward(state)

        if deterministic:
            return action_probs.argmax().item()
        else:
            m = Categorical(action_probs)
            return m.sample().item()

    def get_action_prob(
        self, state: list | tuple | torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Get probability of taking a specific action.

        Args:
            state: Current state as list, tuple, or tensor
            action: Action tensor to get probability for

        Returns:
            Probability of taking the specified action
        """
        action_probs = self.forward(state)
        return action_probs.gather(1, action.unsqueeze(1)).squeeze(1)
