import torch.nn as nn

class net(nn.Module):
    def __init__(self) -> None:
        super(net,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4,2),
            nn.ReLU(),
            nn.Linear(2,1)
        )
    
    def forward(self,x):
        return self.layers(x)

if __name__ == "__main__":
    net1 = net()
    net2 = net()
    tau = 0.1
    def _print(net):
        for param in net.parameters():
            print(param)
    _print(net2)
    for param1,param2 in zip(net1.parameters(),net2.parameters()):
        param2.data.copy_((1-tau)*param1 + tau * param2)
    print("another net")
    _print(net2)