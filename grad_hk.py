import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================================
class encoder_net(nn.Module):

    def __init__(self):
        super(encoder_net,self).__init__()
        self.fc1 = nn.Linear(1,2,bias=False)
        with torch.no_grad():
            self.fc1.weight.fill_(1.0)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

# =======================================
class think_net(nn.Module):

    def __init__(self):
        super(think_net,self).__init__()
        self.fc1 = nn.Linear(2,2,bias=False)
        self.fc2 = nn.Linear(2,1,bias=False)
        with torch.no_grad():
            self.fc1.weight.fill_(1.0)
            self.fc2.weight.fill_(1.0)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# =======================================
def optimize(tnet,z):

    zgrad=[]
    esp = 0.1
    niter = 3
    z.requires_grad_(True)
    z.register_hook(lambda a:zgrad.append(a))

    for t in range(niter):
        y = tnet(z)
        y.backward(torch.tensor([1.]),retain_graph=True)
        print('t ',t,' : z ',z,' zgrad ',zgrad,' y ',y)
        with torch.no_grad():
            z -= esp*zgrad[t-1]
        #z.grad.zero_()

    return z
# =======================================

if __name__=='__main__':

    enet = encoder_net()
    enet.train()
    tnet = think_net()
    tnet.train()

    # this code pass through encoder but passing through
    # encoder break the code, don't know why
    x = torch.tensor([1.0],requires_grad=False)
    x.requires_grad_(False)
    z = enet(x)
    #print('z ',z)

    #z = torch.tensor([1.0,1.0])
    z = optimize(tnet,z)

    print('final z ',z,' z grad ',z.grad)
