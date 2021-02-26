import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
lda = 10000000
# =======================================
class encoder_net(nn.Module):
    def __init__(self):
        super(encoder_net, self).__init__()
        self.fc1 = nn.Linear(1, 2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


# =======================================
class think_net(nn.Module):
    def __init__(self, nhidden=32):
        super(think_net, self).__init__()
        self.fc1 = nn.Linear(2, nhidden, bias=True)
        self.fc2 = nn.Linear(nhidden, nhidden, bias=True)
        self.fc3 = nn.Linear(nhidden, 2, bias=True)
        # with torch.no_grad():
        #    self.fc1.weight.fill_(0.1)
        #    self.fc2.weight.fill_(0.1)
        #    self.fc3.weight.fill_(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


# =======================================
class classifier_net(nn.Module):
    def __init__(self):
        super(classifier_net, self).__init__()
        self.fc1 = nn.Linear(2, 2, bias=True)
        self.fc2 = nn.Linear(2, 2, bias=True)
        # with torch.no_grad():
        #    self.fc1.weight.fill_(0.5)
        #    self.fc2.weight.fill_(1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = torch.softmax(x, dim=1)
        return x


# =======================================
def grad_descend(tnet, z, niter=8, eps=0.1):

    z_list = []
    z.retain_grad()
    z_list.append(z)

    for t in range(niter):
        y = tnet(z_list[t])
        y.backward(torch.ones([ndata, 2]))
        #print('t ',t,' : z ',z_list[t],' zgrad ',z_list[t].grad,' y ',y)
        with torch.no_grad():
            a = z_list[t] - eps * z_list[t].grad
            #print(z_list[t].grad)
            a.requires_grad_(True)
        z_list[t].grad.zero_()
        z_list.append(a)

    return z_list


# =======================================
def print_param(net):

    for param in net.parameters():
        print(param.data)


# =======================================
def prepare_data(ndata):

    x = torch.rand([ndata, 1])
    y = torch.squeeze((x[:,] > 0.5).long())

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

    ndata = 64  # total number of data point, feed in a batch
    nepoch = 10000

    for epoch in range(nepoch):

        x, label = prepare_data(ndata)  # prepare new data every epoch
        position = torch.FloatTensor([[i.item(), i.item()] for i in label])

        z = enet(x)  # encode
        z_list = grad_descend(tnet, z)  # perform gradient descend
        zbatch = z_list[-1]  # feed the final time step z into cnet
        pred = cnet(zbatch)  # classify optimized z
        loss = nn.functional.cross_entropy(pred, label)
        loss = loss + torch.mul(lda, torch.mean((position - zbatch) ** 2))
        if epoch % (nepoch // 1000) == 0:
            print("loss ", loss)
            # plot_tnet()
        loss.backward()
        net_opt.step()  # update weights end2end