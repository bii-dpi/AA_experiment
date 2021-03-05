"""Train RobustNet."""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import *
from data_utils import *
from model_utils import *


torch.autograd.set_detect_anomaly(True)

##################### Constants ########################
SEED = 12345
N_DATA = 64
# TODO: generalize creation of this.
POSITION_DICT = {0: [0 for _ in range(2)],
					1: [1 for _ in range(2)]}
N_EPOCHS = 10000
N_UPDATES = 10
UPDATE_EPS = 0.1


###################### Training ######################
def update_output(updater, encoded_x, y):
	"""Update encoder output."""
	encoded_x.retain_grad()
	encoded_x_list = [encoded_x]
	for curr_iter in range(N_UPDATES):
		y = updater(encoded_x_list[curr_iter])
		y.backward(torch.ones([N_DATA, 1]), retain_graph=True)
		with torch.no_grad():
			delta = UPDATE_EPS * encoded_x_list[curr_iter].grad
			updated_encoded_x = encoded_x_list[curr_iter] - delta
			updated_encoded_x.requires_grad_(True)
		#encoded_x_list[curr_iter].grad.zero_()
		encoded_x_list.append(updated_encoded_x)
	return encoded_x_list[-1]


def get_mse_loss(prediction, y):
	return torch.sum(torch.sub(prediction, y) ** 2)


def get_updater_loss(updated_x, y):
	class_positions = torch.Tensor([POSITION_DICT[entry.item()] for entry in y])
	return get_mse_loss(updated_x, class_positions)


if __name__ == "__main__":
	torch.manual_seed(SEED)

	encoder, updater, classifier = Encoder(), Updater(), Classifier()
	encoder.train(); updater.train(); classifier.train()

	encoder_classifier_optim = optim.Adam(list(encoder.parameters()) + \
											list(classifier.parameters()), lr=0.1)
	updater_optim = optim.Adam(updater.parameters(), lr=0.1)

	for epoch in range(1, N_EPOCHS + 1):
		encoder_classifier_optim.zero_grad()
		updater_optim.zero_grad()

		x, y = prepare_circle_data(N_DATA, torch.device("cpu"))

		encoded_x = encoder(x)
		#print(torch.round(encoded_x))
		updated_x = update_output(updater, encoded_x, y)
		prediction = classifier(updated_x)

		updater_loss = get_updater_loss(encoded_x, y)
		updater_loss.backward(retain_graph=True)
		encoder.zero_grad()
		updater_optim.step()

		encoder_classifier_loss = F.nll_loss(prediction, y)
		encoder_classifier_loss.backward()
		encoder_classifier_optim.step()

		if epoch % 500 == 0 or epoch == 1:
			mean_ec_loss = encoder_classifier_loss.item() / N_DATA
			mean_updater_loss = updater_loss.item() / N_DATA
			print(f"Epoch {epoch}\n"
					f"Updater loss: {mean_updater_loss:.3f}  "
					f"Encoder & classifier loss: {mean_ec_loss:.3f}")
			plot_model_2D_input(updater, torch.device("cpu"), f"Epoch {epoch}")
