"""Script to train updater net on the 'circular-boundary' dataset."""

import time
import torch
import argparse

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from models import UpdaterNet, ClassifierNet
from model_utils import plot_model_2D_input, estimate_time_left, get_updated_input
from data_utils import prepare_circle_data, prepare_linear_data 


def train(args, updater, classifier, prepare_data, device, optimizer, epoch):
	updater.train()
	classifier.train()
	
	optimizer.zero_grad()
		
	x, y = prepare_data(args.batch_size, device)
	x.requires_grad = True
	z = get_updated_input(updater, x, y)
	output = classifier(z)
	loss = F.cross_entropy(output, y)
	loss.backward()
	optimizer.step()

	return loss


def main():
	# Training settings.
	parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
	parser.add_argument("--dataset", type=str, default="linear",
						help="dataset to use (default: linear)")
	parser.add_argument("--batch-size", type=int, default=64,
						help="input batch size for training (default: 64)")
	parser.add_argument("--num-epochs", type=int, default=10000,
						help="number of epochs to train (default: 1000)")
	parser.add_argument("--lr", type=float, default=0.1,
						help="learning rate (default: 0.1)")
	parser.add_argument("--no-cuda", action="store_true", default=True,
						help="disables CUDA training")
	parser.add_argument("--dry-run", action="store_true", default=False,
						help="quickly check a single pass")
	parser.add_argument("--seed", type=int, default=12345,
						help="random seed (default: 12345)")
	parser.add_argument("--log-interval", type=int, default=1000,
						help="interval to print training status (default: 1000)")
	parser.add_argument("--no-plot", action="store_true", default=False,
						help="disables updater net plotting")

	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	train_kwargs = {'batch_size': args.batch_size}
	if use_cuda:
		"""
		cuda_kwargs = {'num_workers': 1,
					   'pin_memory': True,
					   'shuffle': True}
		train_kwargs.update(cuda_kwargs)
		"""
	
	prepare_data = prepare_linear_data if args.dataset == "linear" else prepare_circle_data

	updater = UpdaterNet().to(device)
	classifier = ClassifierNet().to(device)
	optimizer = optim.Adam(list(updater.parameters()) + list(classifier.parameters()), lr=args.lr)
	
	for epoch in range(1, args.num_epochs + 1):
		start = time.process_time()
		loss = train(args, updater, classifier, prepare_data, device, optimizer, epoch)
		time_taken = time.process_time() - start

		if epoch == 1 and args.dry_run:
			break
		if  epoch == 1 or epoch % args.log_interval == 0:
			print(f"Epoch {epoch}")
			estimate_time_left(epoch, args.num_epochs, time_taken)
			if not args.no_plot:
				plot_model_2D_input(updater, device, f"Epoch {epoch}")
			print(f"Loss: {loss.item():.2f}\n")


if __name__ == "__main__":
   main() 
