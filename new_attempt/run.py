"""Script to train updater net on the circular and linear boundary datasets."""

import time
import torch
import argparse

import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from models import UpdaterNet, ClassifierNet
from data_utils import prepare_circle_data, prepare_linear_data, get_class_positions_dict
from model_utils import plot_model_2D_input, estimate_time_left, get_updated_input, get_accuracy


def train(args, updater, class_positions_dict, classifier, prepare_data, device, optimizer):
	updater.train()
	classifier.train()
	
	optimizer.zero_grad()
		
	x, y = prepare_data(args.batch_size, device)
	x.requires_grad = True
	z = get_updated_input(updater, x, y, class_positions_dict, args.num_updates, args.update_lr)
	output = classifier(z)
	loss = F.cross_entropy(output, y)
	loss.backward()
	optimizer.step()
	return loss.item() / args.batch_size, get_accuracy(output, y)


def test(args, updater, class_positions_dict, classifier, prepare_data, device):
	updater.eval()
	classifier.eval()
	#with torch.no_grad():
	x, y = prepare_data(args.batch_size, device)
	x.requires_grad = True
	z = get_updated_input(updater, x, None, class_positions_dict, args.num_updates, args.update_lr)
	output = classifier(z)
	loss = F.cross_entropy(output, y).item() / args.batch_size
	return loss, get_accuracy(output, y)

		
def main():
	# Training settings.
	parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
	parser.add_argument("--dataset", type=str, default="linear",
						help="dataset to use (default: linear)")
	parser.add_argument("--batch-size", type=int, default=64,
						help="input batch size for training (default: 64)")
	parser.add_argument("--num-epochs", type=int, default=10000,
						help="number of epochs to train (default: 10000)")
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
	parser.add_argument("--num-updates", type=int, default=20,
						help="number encoded input update steps (default: 10)")
	parser.add_argument("--update-lr", type=int, default=0.1,
						help="update step learning rate (default: 0.1)")

	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	prepare_data = prepare_linear_data if args.dataset == "linear" else prepare_circle_data

	class_positions_dict = get_class_positions_dict(prepare_data(args.batch_size, device)[1])	

	updater = UpdaterNet().to(device)
	classifier = ClassifierNet().to(device)
	optimizer = optim.Adam(list(updater.parameters()) + list(classifier.parameters()), lr=args.lr)
	
	for epoch in range(1, args.num_epochs + 1):
		start = time.process_time()
		training_loss, training_accuracy = train(args, updater, class_positions_dict, classifier,
													prepare_data, device, optimizer)
		testing_loss, testing_accuracy = test(args, updater, class_positions_dict, classifier,
												prepare_data, device)
		time_taken = time.process_time() - start

		if epoch == 1 and args.dry_run:
			break
		if  epoch == 1 or epoch % args.log_interval == 0:
			print(f"Epoch {epoch}")
			estimate_time_left(epoch, args.num_epochs, time_taken)
			if not args.no_plot:
				plot_model_2D_input(updater, device, f"Epoch {epoch}")
			print(f"Training CE loss: {training_loss:.2f}, accuracy: {training_accuracy:.1f}%")
			print(f"Testing CE loss: {testing_loss:.2f}, accuracy: {testing_accuracy:.1f}%\n")


if __name__ == "__main__":
   main() 
