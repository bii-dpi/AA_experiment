import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# =======================================
class net(nn.Module):
    def __init__(self, nhidden=128):
        super(net, self).__init__()
        self.fc1 = nn.Linear(1,1)
        with torch.no_grad():
            self.fc1.weight.fill_(2.0)
            self.fc1.bias.fill_(1.0)

    def forward(self, x):
        x = self.fc1(x)
        return x

# =======================================
def print_param(param):
    for p in param:
        print('p ',p)
# =======================================
if __name__ == "__main__":

    torch.manual_seed(23839)
    fnet = net()
    fnet.train()

    net_opt = optim.SGD(fnet.parameters(), lr=0.1, momentum=0)
    print_param(fnet.parameters())

    x0 = torch.ones([1,1],requires_grad=True)
    print('x0 ',x0)

    x1 = x0 + fnet(x0)
    print('x1 ',x1)
    x2 = x1 + fnet(x1)
    print('x2 ',x2)

    # x2 = x0 + 2 w x0 + w*w x0 + (w+2)*b
    # dx2/dx0 = 1 + 2w + w*2 = 9 if w=2
    # dx2/dw  = 2 x0 + 2 w x0 + b = 7 if w=2,x0=1,b=1
    # dx2/db  = w + 2 = 4 if w=2

    loss = torch.norm(x2)
    print('loss ',loss)
    loss.backward()

    print('x0 grad ',x0.grad)
    print('x1 grad ',x1.grad)
    print('x2 grad ',x2.grad)
    print('w  grad ',fnet.fc1.weight.grad)
    print('b  grad ',fnet.fc1.bias.grad)
    print('w value ',fnet.fc1.weight)
    print('b value ',fnet.fc1.bias)

    net_opt.step()

    print('after: w  grad ',fnet.fc1.weight.grad)
    print('after: b  grad ',fnet.fc1.bias.grad)
    print('after: w value ',fnet.fc1.weight)
    print('after: b value ',fnet.fc1.bias)
