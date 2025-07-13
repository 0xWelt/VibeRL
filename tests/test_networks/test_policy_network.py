"""
Tests for PolicyNetwork - the policy network implementations.

Tests focus on network-specific functionality without retesting base functionality.
"""

import numpy as np
import pytest
import torch

from viberl.networks.policy_network import PolicyNetwork


class TestPolicyNetwork:
    """Test PolicyNetwork functionality."""

    @pytest.fixture
    def network(self) -> PolicyNetwork:
        """Create PolicyNetwork for testing."""
        return PolicyNetwork(state_size=4, action_size=3, hidden_size=64, num_hidden_layers=2)

    def test_forward_pass_shape(self, network: PolicyNetwork) -> None:
        """Test forward pass produces correct output shape."""
        state = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])

        with torch.no_grad():
            output = network(state)

        assert output.shape == (1, 3)

    def test_output_is_probability_distribution(self, network: PolicyNetwork) -> None:
        """Test output is a valid probability distribution."""
        state = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])

        with torch.no_grad():
            logits = network(state)
            probs = torch.softmax(logits, dim=1)

        # Should sum to 1
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
        # All probabilities should be positive
        assert torch.all(probs > 0)

    def test_different_input_shapes(self, network: PolicyNetwork) -> None:
        """Test network works with different batch sizes."""
        # Single state
        state1 = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])
        output1 = network(state1)
        assert output1.shape == (1, 3)

        # Batch of states
        state_batch = torch.FloatTensor(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
        )
        output_batch = network(state_batch)
        assert output_batch.shape == (3, 3)

    def test_act_method(self, network: PolicyNetwork) -> None:
        """Test act method returns valid action."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        action = network.act(state)

        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_forward_pass(self, network: PolicyNetwork) -> None:
        """Test forward pass works correctly."""
        state = np.array([0.1, 0.2, 0.3, 0.4])

        state_tensor = torch.FloatTensor([state])
        with torch.no_grad():
            output = network(state_tensor)

        assert output.shape == (1, 3)

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
        network = PolicyNetwork(
            state_size=state_size, action_size=action_size, hidden_size=32, num_hidden_layers=1
        )

        state = torch.FloatTensor([[0.1] * state_size])
        output = network(state)

        assert output.shape == (1, action_size)
