"""
Tests for value network implementations.

Tests focus on value network-specific functionality.
"""

import numpy as np
import pytest
import torch

from viberl.networks.value_network import QNetwork, VNetwork


class TestQNetwork:
    """Test QNetwork functionality."""

    @pytest.fixture
    def network(self) -> QNetwork:
        """Create QNetwork for testing."""
        return QNetwork(state_size=4, action_size=3, hidden_size=64, num_hidden_layers=2)

    def test_forward_pass_shape(self, network: QNetwork) -> None:
        """Test forward pass produces correct output shape."""
        state = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])

        output = network(state)

        assert output.shape == (1, 3)

    def test_get_q_values_method(self, network: QNetwork) -> None:
        """Test get_q_values method."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        q_values = network.get_q_values(state)

        assert q_values.shape == (1, 3)
        assert isinstance(q_values, torch.Tensor)

    @pytest.mark.parametrize(
        'state_size,action_size',
        [
            (2, 2),
            (8, 4),
            (16, 8),
        ],
    )
    def test_different_sizes(self, state_size: int, action_size: int) -> None:
        """Test network works with different input/output sizes."""
        network = QNetwork(
            state_size=state_size, action_size=action_size, hidden_size=32, num_hidden_layers=1
        )

        state = torch.FloatTensor([[0.1] * state_size])
        output = network(state)

        assert output.shape == (1, action_size)


class TestVNetwork:
    """Test VNetwork functionality."""

    @pytest.fixture
    def network(self) -> VNetwork:
        """Create VNetwork for testing."""
        return VNetwork(state_size=4, hidden_size=64, num_hidden_layers=2)

    def test_forward_pass_shape(self, network: VNetwork) -> None:
        """Test forward pass produces correct output shape."""
        state = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])

        output = network(state)

        assert output.shape == (1,) or output.shape == (1, 1)

    def test_forward_pass_vnetwork(self, network: VNetwork) -> None:
        """Test VNetwork forward pass works correctly."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        state_tensor = torch.FloatTensor([state])
        output = network(state_tensor)

        assert output.shape == (1,) or output.shape == (1, 1)

    @pytest.mark.parametrize('state_size', [2, 8, 16])
    def test_different_sizes(self, state_size: int) -> None:
        """Test network works with different input sizes."""
        network = VNetwork(state_size=state_size, hidden_size=32, num_hidden_layers=1)

        state = torch.FloatTensor([[0.1] * state_size])
        output = network(state)

        assert output.shape == (1,) or output.shape == (1, 1)
