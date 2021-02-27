from torch.autograd import grad
import torch
import torch.nn as nn

# 1. make net : u = wx+b with w=1,b=0
# 2. x_nxt = x + du/dx = x + G
# 3. L = x_nxt  
# 4. dL/dx, dL/dw, dL/db
#
# compute graph
#
#    w  x
#     \/
#      *
#      |   b
#     wx  /
#      \ /
#       +
#       |
#       U.>.dU/dx=w.>. G=w
#                      |
#                 x    |
#                  \  /
#                   +
#                   |
#                  x_nxt
#                   |
#                   L

class net(nn.Module):

    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,1)
        with torch.no_grad():
            self.fc.weight.fill_(1)
            self.fc.bias.fill_(0)

    def forward(self,x):
        U = self.fc(x)
        print('U ',U)
        # here link computationa graph to gradient
        G = grad(U,x,create_graph=True)[0]  
        print('G ',G)
        return x+G


if __name__=='__main__':

    mover = net()
    x = torch.ones([1,1],requires_grad = True) 
    x_nxt = mover(x)

    L = x_nxt 
    L.backward()

    print('x ',x)
    print('x_nxt ',x_nxt)
    print('L ',L)
    print('=============== ')
    print('x grad ',x.grad)
    print('w grad ',mover.fc.weight.grad)
    print('b grad ',mover.fc.bias.grad)

