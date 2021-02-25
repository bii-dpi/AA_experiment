import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# =======================================
def plot_net(title,net,dim):

    # prepare the grid
    npixels = 32
    axis = np.linspace(-0.2, 1.2, npixels)
    X,Y = np.meshgrid(axis,axis)
    mesh_pairs = np.array(np.meshgrid(axis,axis)).T.reshape(-1,2)

    # evaluate the loss
    with torch.no_grad():
        heights = []
        for xy in mesh_pairs:
            input = torch.unsqueeze(torch.FloatTensor(xy),0)
            h = net(input)
            h = h.tolist()[0][dim]
            heights.append([h])
    heights = np.asarray(heights).reshape(npixels,npixels)

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
    ax.set_zlabel('output')
    plt.title(title)
    
    plt.show()


# =======================================
class flow_net(nn.Module):
    def __init__(self, nhidden=128):
        super(flow_net, self).__init__()

        h1 = 2
        self.fc1 = nn.Linear(2      ,h1)
        #self.fc2 = nn.Linear(nhidden,nhidden)
        #self.fc3 = nn.Linear(nhidden,2)

        self.act1 = nn.Identity()
        #self.act1 = nn.LeakyReLU(0.1)
        #self.act2 = nn.LeakyReLU(0.1)
        #self.act3 = nn.Sigmoid()
        #with torch.no_grad():
        #    self.fc1.weight.fill_(-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        #x = self.fc2(x)
        #x = self.act2(x)
        #x = self.fc3(x)
        #x = self.act3(x)
        return x

# =======================================
def grad_flow(fnet, z, niter=10, eps=0.1):
    z_list = []
    z.retain_grad()
    z_list.append(z)
    ndata = z.shape[0]

    sum_delta=0

    for t in range(niter):
        flow_dir = fnet(z_list[t])
        a = z_list[t] + flow_dir
        a.requires_grad_(True)
        z_list.append(a)
        sum_delta = sum_delta + torch.abs(flow_dir)

    return z_list,sum_delta
# =======================================
def print_param(net):
    for param in net.parameters():
        print(param.data)

# =======================================
def prepare_data(ndata):
    x = torch.rand([ndata, 2])
    x.requires_grad = True
    return x
# =======================================
if __name__ == "__main__":

    torch.manual_seed(23839)
    fnet = flow_net()
    fnet.train()
    title = 'initial fnet dim=0'
    plot_net(title,fnet,dim=0)
    title = 'initial fnet dim=1'
    plot_net(title,fnet,dim=1)

    net_opt = optim.Adam(fnet.parameters(),lr=0.1)

    k = 10
    ndata = 2
    nepoch = 100

    position = torch.zeros([ndata,2])
    x = prepare_data(ndata)

    for epoch in range(nepoch+1):

        net_opt.zero_grad()
        z_list,delta = grad_flow(fnet, x)
        zbatch = z_list[-1]
        mse_loss = torch.mean((position - zbatch) ** 2)
        loss = mse_loss
        if epoch%k==0:
            print(f"Total MSE : {loss.item():.2f}")
            print('zbatch ',zbatch)
            print('ave delta ',torch.sum(torch.abs(delta),dim=1)/ndata)
            title = 'fnet dim 0, epoch '+str(epoch)
            plot_net(title,fnet,dim=0)
            title = 'fnet dim 1, epoch '+str(epoch)
            plot_net(title,fnet,dim=1)
        loss.backward()
        net_opt.step()  # update weights end2end
