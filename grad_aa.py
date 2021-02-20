import torch
import torch.nn as nn


# =======================================
class encoder_net(nn.Module):
    def __init__(self):
        super(encoder_net, self).__init__()
        self.fc1 = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.fc1.weight.fill_(1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


# =======================================
class think_net(nn.Module):
    def __init__(self):
        super(think_net, self).__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.fc1.weight.fill_(1.0)
            self.fc2.weight.fill_(1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


# =======================================
def optimize(tnet, z):
    esp = 0.1
    niter = 3
    z.retain_grad()
    for curr_iter in range(1, niter + 1):
        y = tnet(z)
        y.backward()
        with torch.no_grad():
            z -= esp * z.grad
        print(f"Iteration {curr_iter}, dy/dz: {z.grad}")
        z.grad.zero_()
    return z


# =======================================
if __name__ == "__main__":
    enet = encoder_net()
    enet.train()
    tnet = think_net()
    tnet.train()

    # One example
    x = torch.tensor([1.0])
    z = enet(x)
    z = z.clone().detach()
    z.requires_grad = True
    print(f"Initial f: {z}")
    z = optimize(tnet, z)
    print(f"Final z: {z}")
