import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class UpdateLayer(nn.Module):
    def __init__(self):
        super(UpdateLayer, self).__init__()
        
    def forward(self, x, y):
        pass


class MyLinearLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        import math
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x= torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b

class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        # Encoder layers
        # Updater layers
        self.updater_fc_1 = nn.Linear(2, 16)
        self.updater_act_1 = nn.LeakyReLU(0.1)
        self.updater_fc_2 = nn.Linear(16, 16)
        self.updater_act_2 = nn.LeakyReLU(0.1)
        self.updater_fc_3 = nn.Linear(16, 1)
        # Classifier layers
        self.classifier_fc_1 = nn.Linear(2, 2)

    def plot_updater(self, epoch, device):
        def get_updater_output(x):
            with torch.no_grad():
                updater_output = self.updater_fc_1(x)
                updater_output = self.updater_act_1(updater_output)
                updater_output = self.updater_fc_2(updater_output)
                updater_output = self.updater_act_2(updater_output)
                updater_output = self.updater_fc_3(updater_output)
                return [torch.sigmoid(updater_output).item()]
        npixels = 32
        axis = np.linspace(-0.1, 1.1, npixels)
        X, Y = np.meshgrid(axis, axis)
        mesh_pairs = np.array(np.meshgrid(axis, axis)).T.reshape(-1, 2)
    
        with torch.no_grad():
            Z = []
            for xy in mesh_pairs:
                input = torch.unsqueeze(torch.FloatTensor(xy), 0).to(device)
                z = get_updater_output(input)
                Z.append([z])
        Z = np.asarray(Z).reshape(npixels,npixels)
    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
     
        surf = ax.plot_surface(np.array(X), np.array(Y), np.array(Z), cmap=cm.coolwarm,
                               linewidth=0, antialiased=True)

        ax.set_zlim(np.min(Z) - 1e-2, np.max(Z) + 1e-2)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Updater output')
        plt.title(f"Epoch {epoch}")
        
        plt.show()
    
    def update(self, x, updater_output, n_data, device, positions):
        updater_loss = torch.subtract(updater_output, positions)
        updater_loss.retain_grad()
        updater_loss.backward(torch.ones([n_data, 1]).to(device))
        return x - updater_loss.grad

    def forward(self, x, n_data, device, y=None):
        if y is not None:
            positions = torch.FloatTensor([[i.item()] for i in y]).to(device)
        else:
            positions = [] # Do PCA here.
        x.requires_grad = True
        # Encoder
        # Updater
        updater_output = self.updater_fc_1(x)
        updater_output = self.updater_act_1(updater_output)
        updater_output = self.updater_fc_2(updater_output)
        updater_output = self.updater_act_2(updater_output)
        updater_output = self.updater_fc_3(updater_output)
        updater_output = torch.sigmoid(updater_output)
        x = self.update(x, updater_output, n_data, device, positions)
        # Classifier
        x = self.classifier_fc_1(x)
        return torch.softmax(x, dim=1)