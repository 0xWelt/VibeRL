"""
Tests for common utility functions.

Tests focus on utility functions without retesting complex algorithms.
"""

import numpy as np
import pytest
import torch

from viberl.utils.common import get_device, normalize_returns, set_seed


class TestSetSeed:
    """Test seed setting functionality."""

    def test_set_seed_basic(self):
        """Test basic seed setting functionality."""
        # Test that we can call the function without error
        set_seed(42)
        assert True

    def test_set_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        set_seed(42)
        torch_result1 = torch.rand(5)

        set_seed(42)
        torch_result2 = torch.rand(5)

        # Results should be identical
        assert torch.equal(torch_result1, torch_result2)

    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        torch_result1 = torch.rand(5)

        set_seed(123)
        torch_result2 = torch.rand(5)

        # Results should be different
        assert not torch.equal(torch_result1, torch_result2)


class TestGetDevice:
    """Test device detection functionality."""

    def test_get_device_returns_device(self):
        """Test that get_device returns a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_cpu_or_cuda(self):
        """Test that device is either CPU or CUDA."""
        device = get_device()
        assert device.type in ['cpu', 'cuda']

    @pytest.mark.parametrize('expected_type', ['cpu', 'cuda'])
    def test_device_type_matches_availability(self, expected_type: str) -> None:
        """Test device type matches CUDA availability."""
        device = get_device()
        if expected_type == 'cuda' and not torch.cuda.is_available():
            pytest.skip('CUDA not available')

        assert device.type == expected_type


class TestNormalizeReturns:
    """Test return normalization functionality."""

    def test_normalize_returns_basic(self):
        """Test basic return normalization."""
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_returns(returns)

        # Should have zero mean
        assert abs(np.mean(normalized)) < 1e-6
        # Should have unit variance (approximately)
        assert abs(np.std(normalized) - 1.0) < 1e-6

    def test_normalize_returns_empty(self):
        """Test normalization with empty array."""
        returns = np.array([])
        normalized = normalize_returns(returns)

        assert len(normalized) == 0
        assert isinstance(normalized, np.ndarray)

    def test_normalize_returns_single_value(self):
        """Test normalization with single value."""
        returns = np.array([5.0])
        normalized = normalize_returns(returns)

        # Single value should result in zero (since mean = value)
        assert normalized[0] == 0.0

    def test_normalize_returns_constant(self):
        """Test normalization with constant values."""
        returns = np.array([3.0, 3.0, 3.0])
        normalized = normalize_returns(returns)

        # Constant values should all become zero
        assert np.allclose(normalized, 0.0)

    def test_normalize_returns_negative_values(self):
        """Test normalization with negative values."""
        returns = np.array([-5.0, -3.0, -1.0, 1.0, 3.0])
        normalized = normalize_returns(returns)

        # Should still have zero mean
        assert abs(np.mean(normalized)) < 1e-6
        # Should have unit variance
        assert abs(np.std(normalized) - 1.0) < 1e-6

    @pytest.mark.parametrize(
        'returns',
        [
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, 0.0, 1.0]),
            np.array([0.5, 1.5, 2.5, 3.5]),
            np.array([100.0, 200.0, 300.0]),
        ],
    )
    def test_normalize_returns_various_inputs(self, returns: np.ndarray) -> None:
        """Test normalization with various input arrays."""
        normalized = normalize_returns(returns)

        # Should preserve shape
        assert normalized.shape == returns.shape
        # Should be numeric
        assert np.all(np.isfinite(normalized))


def test_utils_import():
    """Test that utils module can be imported."""
    from viberl.utils import common

    assert hasattr(common, 'set_seed')
    assert hasattr(common, 'get_device')
    assert hasattr(common, 'normalize_returns')
