import torch
import torch.nn as nn
import numpy as np
class VAE_actiongenerate(nn.Module):
    def __init__(self) -> None:
        super(VAE_actiongenerate,self).__init__()
        self.encodermu = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,4),
            # nn.ReLU(),
            # nn.Linear(4,2)
        )
        self.encodersigma = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,4),
            # nn.ReLU(),
            # nn.Linear(4,2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.Tanh()
        )
        # self.zmu = torch.tensor([1.0,1.0,1.0,1.0]).cuda()
        # self.zsigma = torch.tensor([1.0,1.0]).cuda()
    
    def forward(self,state:np.ndarray):
        if isinstance(state,np.ndarray):
            state = torch.from_numpy(state)
        state = state.cuda().to(torch.float32)
        mu = self.encodermu(state)
        sigma = self.encodersigma(state)
        sample_data = torch.normal(torch.ones(mu.shape).to(torch.float32).cuda(),torch.ones(mu.shape).to(torch.float32).cuda()).to(torch.float32)
        sample_result = mu + sample_data * sigma
        return self.decoder(sample_result)
import gym
import d4rl
def train():
    net = VAE_actiongenerate()
if __name__ == "__main__":
    vaenet = VAE_actiongenerate().cuda()
    states = np.random.random((5,4))
    result = vaenet(states)
    print(result)
