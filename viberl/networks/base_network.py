import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    """Base neural network architecture for RL agents."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_hidden_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Build network layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        self.backbone = nn.Sequential(*layers)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the shared backbone.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Processed tensor of shape (batch_size, hidden_size)
        """
        return self.backbone(x)

    def init_weights(self) -> None:
        """Initialize network weights using Xavier initialization.

        Uses Xavier uniform initialization for linear layers and zeros for biases.
        This helps with stable gradient flow during training and prevents
        vanishing/exploding gradients.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
