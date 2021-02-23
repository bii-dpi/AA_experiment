import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
lda = 0.01 
# =======================================
class encoder_net(nn.Module):
    def __init__(self):
        super(encoder_net, self).__init__()

    def forward(self, x):
        return x


# =======================================
class think_net(nn.Module):
    def __init__(self, nhidden=4):
        super(think_net, self).__init__()
        self.fc1 = nn.Linear(2, nhidden)
        self.fc2 = nn.Linear(nhidden, 1)
        with torch.no_grad():
           self.fc1.weight.fill_(0.5)
           self.fc2.weight.fill_(1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


# =======================================
class classifier_net(nn.Module):
    def __init__(self):
        super(classifier_net, self).__init__()
        self.fc1 = nn.Linear(2, 2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.softmax(x, dim=1)
        return x


# =======================================
def grad_descend(tnet, z, niter=8, eps=0.1):
    z_list = []
    z.retain_grad()
    z_list.append(z)

    for t in range(niter):
        y = tnet(z_list[t])
        y.backward(torch.ones([ndata, 1]))
        #print('t ',t,' : z ',z_list[t],' zgrad ',z_list[t].grad,' y ',y)
        with torch.no_grad():
            a = z_list[t] - eps * z_list[t].grad
            #print(z_list[t].grad)
            a.requires_grad_(True)
        z_list[t].grad.zero_()
        z_list.append(a)

    return z_list


# =======================================
def plot_tnet(epoch):
    with torch.no_grad():
        tnet.eval()
        X = []
        Y = []
        heights = []
        axis = np.linspace(-0.2, 1.2, 256)
        for x1 in axis:
            for x2 in axis:
                X.append(x1)
                Y.append(x2)
                curr_input = torch.FloatTensor([x1, x2])
                heights.append([tnet(curr_input).item()])
        tnet.train()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
 
    # Plot the surface.
    surf = ax.plot_surface(np.array(X), np.array(Y), np.array(heights), #cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    
    print('z range ',np.min(heights),':',np.max(heights))

    # Customize the z axis.
    zmin = np.min(heights)-1e-3
    zmax = np.max(heights)+1e-3
    ax.set_zlim(zmin, zmax)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('tnet output')
    plt.title(f"Epoch {epoch}")
    
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()


# =======================================
def print_param(net):
    for param in net.parameters():
        print(param.data)


# =======================================
def prepare_data(ndata):
    x = torch.rand([ndata, 2])
    y = torch.squeeze(torch.sum(x ** 2, axis=1) < np.sqrt(2 / np.pi)).long()

    return x, y


# =======================================
if __name__ == "__main__":
    torch.manual_seed(263839)
    enet = encoder_net()
    tnet = think_net()
    cnet = classifier_net()
    enet.train()
    tnet.train()
    cnet.train()

    end2end_param = (
        list(enet.parameters()) + list(tnet.parameters()) + list(cnet.parameters())
    )
    net_opt = optim.Adam(end2end_param, weight_decay=0.01)

    ndata = 64
    nepoch = 10000

    for epoch in range(nepoch+1):
        x, label = prepare_data(ndata)
        position = torch.FloatTensor([[i.item(), i.item()] for i in label])

        z = enet(x)
        z.requires_grad = True
        z_list = grad_descend(tnet, z)
        zbatch = z_list[-1]
        pred = cnet(zbatch)
        ce_loss = nn.functional.cross_entropy(pred, label)
        mse_loss = torch.mul(lda, torch.mean((position - zbatch) ** 2))
        loss = ce_loss + mse_loss
        if epoch % 500 == 0:
            print(f"Total: {loss.item():.2f} || CE: {ce_loss.item():.2f}, MSE: {mse_loss.item()/lda:.2f}")
            plot_tnet(epoch)
        loss.backward()
        net_opt.step()  # update weights end2end
    plot_tnet(epoch)
