"""Utility functions to generate data."""

import torch
import numpy as np

from typing import Tuple


def prepare_data(n_data: int, dim: int) -> torch.Tensor:
    """Prepares n_data of dimension dim."""
    return torch.rand([n_data, dim])


def is_too_imbalanced(y: torch.Tensor, neg_cutoff: float=0.3) -> bool:
    """Checks if dataset is too imbalanced."""
    return 1 - torch.sum(y) / list(y.shape)[0] < neg_cutoff


def prepare_linear_data(n_data: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares linearly-separable data."""
    x = prepare_data(n_data, 1)
    y = torch.squeeze((x > 0.5).long())
    while is_too_imbalanced(y):
        x = prepare_data(n_data, 1)
        y = torch.squeeze((x > 0.5).long())
    return x.to(device), y.to(device)


def prepare_circle_data(n_data: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares data separated by a circular boundary."""
    x = prepare_data(n_data, 2)
    y = torch.squeeze(torch.sum(x ** 2, axis=1) < (2.0 / np.pi)).long()
    while is_too_imbalanced(y):
        x = prepare_data(n_data, 2)
        y = torch.squeeze(torch.sum(x ** 2, axis=1) < (2.0 / np.pi)).long()
    return x.to(device), y.to(device)
 
