"""Updater-net model class definition."""

import torch
import torch.nn as nn


class UpdaterNet(nn.Module):
    """Updater net definition updates inputs wrt to a surrogate loss."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)

class ClassifierNet(nn.Module):
    """Classifies updated input."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.fc1(x), dim=1)
