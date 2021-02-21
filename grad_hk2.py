import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =======================================
class encoder_net(nn.Module):

    def __init__(self):
        super(encoder_net,self).__init__()
        self.fc1 = nn.Linear(1,2,bias=True)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x
# =======================================
class think_net(nn.Module):

    def __init__(self):
        super(think_net,self).__init__()
        self.fc1 = nn.Linear(2 ,32,bias=True)
        self.fc2 = nn.Linear(32,32,bias=True)
        self.fc3 = nn.Linear(32,1 ,bias=True)
        #with torch.no_grad():
        #    self.fc1.weight.fill_(0.1)
        #    self.fc2.weight.fill_(1.0)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        return x
# =======================================
class classifier_net(nn.Module):
    def __init__(self):
        super(classifier_net,self).__init__()
        self.fc1 = nn.Linear(2,2,bias=False)
        self.fc2 = nn.Linear(2,2,bias=False)
        #with torch.no_grad():
        #    self.fc1.weight.fill_(0.5)
        #    self.fc2.weight.fill_(1.0)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# =======================================
def grad_descend(tnet,z,niter=20,eps=0.01):

    z_list = []
    z.retain_grad()
    z_list.append(z)

    for t in range(niter):
        y = tnet(z_list[t])
        y.backward(torch.ones([ndata,1]))
        #print('t ',t,' : z ',z_list[t],' zgrad ',z_list[t].grad,' y ',y)
        with torch.no_grad():
            a = z_list[t] - eps*z_list[t].grad
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
    
    x = torch.rand([ndata,1])
    y = torch.squeeze((x[:,]>0.5).long())

    return x,y
# =======================================

if __name__=='__main__':

    torch.manual_seed(172839)
    enet = encoder_net()
    tnet = think_net()
    cnet = classifier_net()
    enet.train()
    tnet.train()
    cnet.train()

    end2end_param = list(enet.parameters()) + \
                    list(tnet.parameters()) + \
                    list(cnet.parameters())
    net_opt = optim.Adam(end2end_param)

    ndata = 128  # total number of data point, feed in a batch
    nepoch = 100000

    for epoch in range(nepoch):

        x,label = prepare_data(ndata) # prepare new data every epoch
        
        z = enet(x)                   # encode
        z_list = grad_descend(tnet,z) # perform gradient descend
        #print('z list first ',z_list)

        zbatch = z_list[-1]           # feed the final time step z into cnet
        #print('zbatch ',zbatch)
        pred = cnet(zbatch)           # classify optimized z
        #print('pred ',pred,'\n label ',label)

        loss = nn.functional.cross_entropy(pred,label)
        if epoch%(nepoch//100)==0:
            print('loss ',loss)
            # plot tnet surface here
            # plot_tnet()
        loss.backward()
        net_opt.step()                # update weights end2end
        
        #print('enet parameters =============== ')
        #print_param(enet)
        #print('tnet parameters =============== ')
        #print_param(tnet)
        #print('cnet parameters =============== ')
        #print_param(cnet)

