import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder_fc1 = nn.Linear(2, 2)
		self.encoder_fc2 = nn.Linear(2, 2)

	def forward(self, x):
		x = torch.sigmoid(self.encoder_fc1(x))
		# Encoded output should be within unit hypercube.
		return torch.sigmoid(self.encoder_fc2(x))


class Updater(nn.Module):
	def __init__(self):
		super().__init__()
		self.updater_fc1 = nn.Linear(2, 4)
		self.updater_fc2 = nn.Linear(4, 1)
		# Initialize the parameters to give an initial
		# output of 1 for all inputs.
		#with torch.no_grad():
		   #self.updater_fc1.weight.fill_(3)
		   #self.updater_fc2.weight.fill_(3)

	def forward(self, x):
		x = torch.sigmoid(self.updater_fc1(x))
		return torch.sigmoid(self.updater_fc2(x))


class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.classifier_fc1 = nn.Linear(2, 2)
		self.classifier_fc2 = nn.Linear(2, 2)

	def forward(self, x):
		# Will take in updated encoder output.
		x = torch.sigmoid(self.classifier_fc1(x))
		x = self.classifier_fc2(x)
		return F.log_softmax(x, dim=1)

