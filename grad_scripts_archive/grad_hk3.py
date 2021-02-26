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
    def __init__(self, nhidden=128):
        super(think_net, self).__init__()
        self.fc1 = nn.Linear(2,       nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, 1)
        self.act1 = nn.LeakyReLU(0.1)
        self.act2 = nn.LeakyReLU(0.1)
        #with torch.no_grad():
        #   self.fc1.weight.fill_(1e3)
        #   self.fc2.weight.fill_(1e-3)
        #   self.fc3.weight.fill_(1e3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
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
def grad_descend(tnet, z, niter=2, eps=0.1):
    z_list = []
    z.retain_grad()
    z_list.append(z)

    sum_delta=0

    for t in range(niter):
        y = tnet(z_list[t])
        y.backward(torch.ones([ndata, 1]))
        #print('t ',t,' : z ',z_list[t],' zgrad ',z_list[t].grad,' y ',y)
        with torch.no_grad():
            delta = eps * z_list[t].grad
            a = z_list[t] - delta
            a.requires_grad_(True)
        z_list[t].grad.zero_()
        z_list.append(a)
        sum_delta = sum_delta + torch.abs(delta)

    return z_list,sum_delta

# =======================================
# this function checks that plot is correct
def add_test(input):
    r = input[0][0]+input[0][1]
    r = torch.unsqueeze(torch.unsqueeze(r,0),0)
    return r
# =======================================
def plot_net(title,net):

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
            h = h.tolist()[0][0]
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
def print_param(net):
    for param in net.parameters():
        print(param.data)

# =======================================
def prepare_data(ndata):
    x = torch.rand([ndata, 2])
    # error in label
    y = torch.squeeze(torch.sum(x ** 2, axis=1) < (2.0 / np.pi)).long()

    return x, y

# =======================================
if __name__ == "__main__":

    plot_net('pls visually check plot',add_test)

    torch.manual_seed(267839)
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

    ndata = 256
    nepoch = 1000

    for epoch in range(nepoch+1):
        x, label = prepare_data(ndata)
        position = torch.FloatTensor([[i.item(), i.item()] for i in label])
        data_balance = torch.sum(label)*1.0/ndata
        assert torch.abs(data_balance-0.5)<0.2,'data imbalance, cannot train'

        z = enet(x)
        z.requires_grad = True
        z_list,delta = grad_descend(tnet, z)
        zbatch = z_list[-1]
        pred = cnet(zbatch)
        ce_loss = nn.functional.cross_entropy(pred, label)
        mse_loss = torch.mul(lda, torch.mean((position - zbatch) ** 2))
        #loss = ce_loss + mse_loss
        loss =  mse_loss
        if epoch % (nepoch//5) == 0:
            print(f"Total: {loss.item():.2f} || CE: {ce_loss.item():.2f}, MSE: {mse_loss.item()/lda:.2f}")
        print('ave delta ',torch.sum(torch.abs(delta))/ndata)
        title = 'tnet : '+str(epoch)
        plot_net(title,tnet)
        loss.backward()
        net_opt.step()  # update weights end2end
    print('done with opt ================ ')
    plot_net('cnet',cnet)
