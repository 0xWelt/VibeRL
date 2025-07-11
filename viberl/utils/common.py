import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        PyTorch device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize_returns(returns: np.ndarray) -> np.ndarray:
    """
    Normalize returns to zero mean and unit variance.

    Args:
        returns: Array of returns

    Returns:
        Normalized returns
    """
    if len(returns) == 0:
        return returns

    mean = np.mean(returns)
    std = np.std(returns)

    if std == 0:
        return returns - mean

    return (returns - mean) / (std + 1e-8)
