"""Utility functions to debug models."""

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib import cm
from typing import Dict, Iterable, Callable
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from data_utils import get_class_positions, get_nearest_class_positions


def plot_model_2D_input(model: nn.Module, device: torch.device, title: str, npixels: int = 32) -> None:
	"""Visualize models with two-dimensional inputs."""
	# Prepare input mesh grid. Inputs are assumed to exist in [0, 1]
	axis = np.linspace(-0.1, 1.1, npixels)
	X, Y = np.meshgrid(axis, axis)
	mesh_pairs = np.array(np.meshgrid(axis, axis)).T.reshape(-1, 2)

	# Prepare outputs.
	with torch.no_grad():
		Z = []
		for xy in mesh_pairs:
			curr_input = torch.unsqueeze(torch.FloatTensor(xy), 0).to(device)
			z = model(curr_input)
			Z.append([z])

	# Reshape flat output liist to conform with mesh grid.
	Z = np.asarray(Z).reshape(npixels,npixels)

	# Initialize figure and plot surface.
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(np.array(X), np.array(Y), np.array(Z),
							cmap=cm.coolwarm, linewidth=0, antialiased=True)

	ax.set_zlim(np.min(Z) - 1e-2, np.max(Z) + 1e-2)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
	ax.set_xlabel("x1")
	ax.set_ylabel("x2")
	ax.set_zlabel("Output")
	plt.title(title)

	plt.show()


def plot_model_1D_input() -> None:
	# TODO: complete definition to plot encoder, classifier.
	pass


def estimate_time_left(epoch: int, num_epochs: int, time_taken: float) -> None:
	"""Print estimated time left."""
	# XXX: this method does not seem to work properly.
	print(f"{epoch / num_epochs * 100:.1f}% done; "
			f"time left: {((num_epochs - epoch) / epoch) * time_taken * 100:.2f}s")


def get_accuracy(output, y):
	"""Get accuracy of a softmax-ed output."""
	return ((torch.sum(torch.argmax(output, dim=1) == y) * 100) / len(y)).item()

