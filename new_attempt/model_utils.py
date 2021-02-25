"""Utility functions to debug models."""

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib import cm
from typing import Dict, Iterable, Callable
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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


def plot_model(model: nn.Module, device: torch.device, title: str, npixels: int = 32) -> None:
    """Visualize models ."""
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


def estimate_time_left(epoch: int, num_epochs: int, time_taken: float) -> None:
    """Print estimated time left."""
    print(f"{epoch / num_epochs * 100:.1f}% done; "
            f"time left: {((num_epochs - epoch) / epoch) * time_taken:.3f}s")