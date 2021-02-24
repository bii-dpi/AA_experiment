# This script verifies that regardless even if the loss function is indirectly-
# related to tnet, tnet itself does not update unless y.backward() is called.
# In effect, PyTorch seems to "ignore" tnet since it does not produce output
# directly involved with the loss.

# I noticed this when working on my own fix to this issue, in which I was trying
# to make the network "more end-to-end".

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


# =======================================
class think_net(nn.Module):
    def __init__(self, nhidden=128):
        super(think_net, self).__init__()
        self.fc1 = nn.Linear(2,       nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, 1)
        self.act1 = nn.LeakyReLU(0.1)
        self.act2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


# =======================================
def grad_descend(tnet, z, niter=2, eps=0.1):
    z_list = []
    z.retain_grad()
    z_list.append(z)
    ndata = z.shape[0]
    all_y = []
    sum_delta=0

    for t in range(niter):
        y = tnet(z_list[t])
        all_y.append(y)
        y.backward(torch.ones([ndata, 1]))
        #print('t ',t,' : z ',z_list[t],' zgrad ',z_list[t].grad,' y ',y)
        with torch.no_grad():
            delta = eps * z_list[t].grad
            a = z_list[t] - delta
            a.requires_grad_(True)
        z_list[t].grad.zero_()
        z_list.append(a)
        sum_delta = sum_delta + torch.abs(delta)

    return z_list[-1]

def dummy_grad_descend(tnet, z, niter=2, eps=0.1):
    return z


# =======================================
def prepare_data(ndata):
    x = torch.rand([ndata, 2])
    x.requires_grad = True
    return x

def are_equal(init_params, final_params):
    result = [np.all(init_params[i] == final_params[i]) for i in range(len(init_params))]
    return np.all(np.array(result) == True)

def train(grad_descent_function):
    torch.manual_seed(267839)
    tnet = think_net()
    tnet.train()

    net_opt = optim.Adam(tnet.parameters(), weight_decay=0.01)

    ndata = 4
    nepoch = 100

    position = torch.zeros([ndata,2])
    x = prepare_data(ndata)

    for epoch in range(1, nepoch+1):
        if epoch == 1:
            init_params = [tensor.clone().detach().numpy() for tensor in tnet.parameters()]
        zbatch  = grad_descent_function(tnet, x)
        mse_loss = torch.mean((position - zbatch) ** 2)
        loss = mse_loss
        loss.backward()
        net_opt.step()
        if epoch == nepoch:
            final_params = [tensor.clone().detach().numpy() for tensor in tnet.parameters()]
    
    print(f"Initial and final parameters are equal: {are_equal(init_params, final_params)}")

# =======================================
if __name__ == "__main__":
    print("y.backward() not called")
    train(dummy_grad_descend)

    print("y.backward() called")
    train(grad_descend)
    