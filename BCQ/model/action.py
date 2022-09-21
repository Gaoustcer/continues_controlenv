import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
class actionnet(nn.Module):
    def __init__(self,state_size = 4,action_size = 2) -> None:
        super(actionnet,self).__init__()
        self.actiondecieison = nn.Sequential(
            nn.Linear(state_size,4),
            nn.ReLU(),
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,action_size)
        )
        self.N = 128
        self.actionoutput = nn.Tanh()
        self.noisegenerator = Normal(0,1)
    def forward(self,states):
        if isinstance(states,np.ndarray):
            states = torch.from_numpy(states).cuda().to(torch.float32)
        action_from_state = self.actiondecieison(states)
        action_from_noise = self.noisegenerator.sample(action_from_state.shape).cuda().to(torch.float32)/self.N
        return self.actionoutput(action_from_noise + action_from_state)

        # return 2 * (self.actiondecieison(states) - 0.5)
