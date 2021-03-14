import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_OLD_METHOD = True
POSITION_DICT = {0: [0 for _ in range(1)], 1: [1 for _ in range(1)]}


# ======================================================
class Updater(nn.Module):
	def __init__(self, h=10):
		super().__init__()
		self.fc1 = nn.Linear(1,h)
		self.fc2 = nn.Linear(h,h)
		self.fc3 = nn.Linear(h,1)

	def forward(self,x):
		x = self.fc1(x)
		x = torch.sigmoid(x)
		x = self.fc2(x)
		x = torch.sigmoid(x)
		x = self.fc3(x)
		x = torch.sigmoid(x)
		return x


# ======================================================
def update_output(updater_net, init_x):
	eps = 1
	x0 = init_x
	x0.retain_grad()
	batch_size = x0.shape[0]

	if USE_OLD_METHOD:
		y0 = updater_net(x0)
		y0.backward(torch.ones([batch_size,1]))
		x1 = x0 - x0.grad
		x1.retain_grad()

		y1 = updater_net(x1)
		y1.backward(torch.ones([batch_size,1]))
		x2 = x1 - x1.grad
	else:
		x1 = x0 - eps * torch.autograd.grad(
			outputs=updater_net(x0),
			inputs=x0,
			grad_outputs=torch.ones([batch_size,1]),
			create_graph=True, # If false, gradients for parameter backprop not computer
			retain_graph=True, # If false, get error as cannot access gradients in backprop.
			only_inputs=True, # No need to return any other gradients apart from dydx.
		)[0]
		# x1.retain_grad()  # No need to retain grad whe grad() is used.
		x2 = x1 - eps*torch.autograd.grad(
			outputs=updater_net(x1),
			inputs=x1,
			grad_outputs=torch.ones([batch_size,1]),
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		# x2.retain_grad()
	return x2


# ======================================================
def get_updater_loss(updated_x, y):
	class_positions = torch.Tensor([POSITION_DICT[entry.item()] for entry in y])
	return torch.sum(torch.sub(updated_x, class_positions) ** 2)


# ======================================================
def print_net(t,net):
	steps = 10
	grid = torch.linspace(0,1,steps)

	for x in grid:
		x = torch.unsqueeze(x, dim=0)
		y = net(x)
		print(x,' ',y)

# ======================================================
if __name__=='__main__':
	torch.manual_seed(1234)
	nepoch = 1000
	batch_size = 100

	net = Updater()
	optimizer = optim.Adam(net.parameters(), lr=0.1)

	x = torch.rand([batch_size, 1], requires_grad=True)
	y = (x > 0.5).long()

	for i in range(1, nepoch + 1):
		optimizer.zero_grad()

		net.train()

		z = update_output(net, x)

		loss = get_updater_loss(z, y)
		loss.backward()
		optimizer.step()

		if i % 100 == 0 or i == 1:
			print(f'Epoch {i}')
			print(loss / batch_size)
			net.eval()
			print_net(i,net)
			print('\n')

"""
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)[source]
Computes and returns the sum of gradients of outputs w.r.t. the inputs.

grad_outputs should be a sequence of length matching output containing the pre-computed gradients w.r.t. each of the outputs. If an output doesn’t require_grad, then the gradient can be None).

If only_inputs is True, the function will only return a list of gradients w.r.t the specified inputs. If it’s False, then gradient w.r.t. all remaining leaves will still be computed, and will be accumulated into their .grad attribute.


Parameters:
outputs (sequence of Tensor) – outputs of the differentiated function.
inputs (sequence of Tensor) – Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).
grad_outputs (sequence of Tensor) – Gradients w.r.t. each output. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable for all grad_tensors, then this argument is optional. Default: None.
retain_graph (bool, optional) – If False, the graph used to compute the grad will be freed. Note that in nearly all cases setting this option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value of create_graph.
create_graph (bool, optional) – If True, graph of the derivative will be constructed, allowing to compute higher order derivative products. Default: False.

"""
