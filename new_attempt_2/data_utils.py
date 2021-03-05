"""Utility functions to generate data."""

import torch
import numpy as np

from typing import Tuple, List


def prepare_data(n_data: int, dim: int) -> torch.Tensor:
	"""Prepares n_data of dimension dim."""
	return torch.rand([n_data, dim])


def is_too_imbalanced(y: torch.Tensor, neg_cutoff: float=0.3) -> bool:
	"""Checks if dataset is too imbalanced."""
	return 1 - torch.sum(y) / list(y.shape)[0] < neg_cutoff


def prepare_linear_data(n_data: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Prepares linearly-separable data."""
	x = prepare_data(n_data, 2)
	y = torch.squeeze((torch.sum(x, axis=1) > 1).long())
	while is_too_imbalanced(y):
		x = prepare_data(n_data, 2)
		y = torch.squeeze((torch.sum(x, axis=1) > 1).long())
	return x.to(device), y.to(device)


def prepare_circle_data(n_data: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Prepares data separated by a circular boundary."""
	x = prepare_data(n_data, 2)
	y = torch.squeeze(torch.sum(x ** 2, axis=1) < (2.0 / np.pi)).long()
	while is_too_imbalanced(y):
		x = prepare_data(n_data, 2)
		y = torch.squeeze(torch.sum(x ** 2, axis=1) < (2.0 / np.pi)).long()
	return x.to(device), y.to(device)


def get_class_positions(class_positions_dict: dict, y: torch.Tensor) -> torch.Tensor:
	"""Get class positions given example targets."""
	return torch.Tensor([[class_positions_dict[entry.item()]] for entry in y])


def get_nearest_class_position(class_positions_dict: dict, output: torch.Tensor) -> List:
	"""Get nearest class position to current updater output."""
	min_dist = np.Inf
	nearest_class = None
	for curr_class, curr_position in class_positions_dict.items():
		curr_dist = torch.sum((output - curr_position) ** 2)
		if curr_dist < min_dist:
			min_dist = curr_dist
			nearest_class = curr_class
	return class_positions_dict[nearest_class]


def get_nearest_class_positions(class_positions_dict: dict, updater_output: torch.Tensor) -> torch.Tensor:
	"""Get nearest class position for each evaluation example."""
	return torch.Tensor([[get_nearest_class_position(class_positions_dict, output)] for output in updater_output])

