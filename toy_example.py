import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


##################### Constants ########################
N_DATA = 64
SEED = 12345
POSITION_DICT = {0: [0], 1:[1]}
N_EPOCHS = 10000


################# RobustNet components #################
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
		self.updater_fc1 = nn.Linear(2, 2)
		self.updater_fc2 = nn.Linear(2, 2)
		
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


################# Data preparation ##################
def prepare_data(n_data, dim):
	"""Prepares n_data of dimension dim."""
	return torch.rand([n_data, dim])


def is_too_imbalanced(y, neg_cutoff=0.3):
	"""Checks if dataset is too imbalanced."""
	return 1 - torch.sum(y) / list(y.shape)[0] < neg_cutoff


def prepare_circle_data(n_data):
	"""Prepares data separated by a circular boundary."""
	x = prepare_data(n_data, 2)
	y = torch.squeeze(torch.sum(x ** 2, axis=1) < (2.0 / np.pi)).long()
	while is_too_imbalanced(y):
		x = prepare_data(n_data, 2)
		y = torch.squeeze(torch.sum(x ** 2, axis=1) < (2.0 / np.pi)).long()
	return x, y


###################### Training ######################
def get_mse_loss(prediction, y):
	return torch.sum(torch.sub(prediction, y) ** 2)


def get_updater_loss(encoded_x, y):
	class_positions = torch.Tensor([POSITION_DICT[entry.item()] for entry in y])
	return get_mse_loss(encoded_x, class_positions)	


if __name__ == "__main__":
	torch.manual_seed(SEED)
	
	encoder = Encoder()
	updater = Updater()
	classifier = Classifier()
	
	encoder.train()
	updater.train()
	classifier.train()

	optimizer = optim.Adam(list(encoder.parameters()) + \
							list(updater.parameters()) + \
							list(classifier.parameters()), lr=0.1)

	for epoch in range(1, N_EPOCHS + 1):
		optimizer.zero_grad()

		x, y = prepare_circle_data(N_DATA)
		
		encoded_x = encoder(x)
		updated_x = updater(encoded_x)
		prediction = classifier(updated_x)
		
		updater_loss = get_updater_loss(encoded_x, y)
		classifier_loss = F.nll_loss(prediction, y)

		updater_loss.backward(retain_graph=True)
		classifier_loss.backward()

		optimizer.step()

		if epoch % 500 == 0 or epoch == 1:
			mean_classifier_loss = classifier_loss.item() / N_DATA
			mean_updater_loss = updater_loss.item() / N_DATA
			print(f"Epoch {epoch}\n"
					f"Updater loss: {mean_updater_loss:.3f}  "
					f"Classifier loss: {mean_classifier_loss:.3f}")
