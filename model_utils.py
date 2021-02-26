"""Utility functions to debug models."""

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib import cm
from typing import Dict, Iterable, Callable
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from data_utils import get_class_positions, get_nearest_class_positions


class VerboseExecution(nn.Module):
	"""A wrapper to display model dimensionality."""
	# From https://bit.ly/3knU5U5 
	def __init__(self, model: nn.Module):
		super().__init__()
		self.model = model

		# Register a hook for each layer
		for name, layer in self.model.named_children():
			layer.__name__ = name
			layer.register_forward_hook(
				lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
			)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)


class FeatureExtractor(nn.Module):
	"""A wrapper to return model features after a forward pass."""
	# From https://bit.ly/3knU5U5 
	def __init__(self, model: nn.Module, layers: Iterable[str]):
		super().__init__()
		self.model = model
		self.layers = layers
		self._features = {layer: torch.empty(0) for layer in layers}

		for layer_id in layers:
			layer = dict([*self.model.named_modules()])[layer_id]
			layer.register_forward_hook(self.save_outputs_hook(layer_id))

	def save_outputs_hook(self, layer_id: str) -> Callable:
		def fn(_, __, output):
			self._features[layer_id] = output
		return fn

	def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		_ = self.model(x)
		return self._features


def plot_model_2D_input(model: nn.Module, device: torch.device, title: str, npixels: int = 32) -> None:
	"""Visualize models with two-dimensional inputs."""
	# Prepare input mesh grid.
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
			

def get_updated_input(updater, x, y, class_positions_dict, num_updates, update_lr):
	""""Something."""
	### XXX: statistics concerning delta, x_list are not used. Consider deletion.
	x_list = []
	x.retain_grad()
	x_list.append(x)

	sum_delta = 0
	
	if y is not None:
		class_positions = get_class_positions(class_positions_dict, y)

	for t in range(num_updates):
		updater_output = updater(x_list[t])
		if y is None:
			class_positions = get_nearest_class_positions(class_positions_dict, updater_output)
		updater_loss = (updater_output - class_positions)  ** 2
		updater_loss.backward(torch.ones([list(x.shape)[0], 1]))
		#print('t ',t,' : x ',x_list[t],' xgrad ',x_list[t].grad,' y ',y)
		with torch.no_grad():
			delta = update_lr * x_list[t].grad
			a = x_list[t] - delta
			a.requires_grad_(True)
		x_list[t].grad.zero_()
		x_list.append(a)
		sum_delta = sum_delta + torch.abs(delta)
	return updater(x_list[-1])


def get_accuracy(output, y):
	"""Get accuracy of a softmax-ed output."""
	return ((torch.sum(torch.argmax(output, dim=1) == y) * 100) / len(y)).item()

