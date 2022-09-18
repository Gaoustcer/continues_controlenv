import torch.nn as nn
import numpy as np
import torch
class actionnet(nn.Module):
    def __init__(self,state_size = 4,action_size = 2) -> None:
        super(actionnet,self).__init__()
        self.actiondecieison = nn.Sequential(
            nn.Linear(state_size,4),
            nn.ReLU(),
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,action_size),
            nn.Sigmoid()
        )
    def forward(self,states):
        if isinstance(states,np.ndarray):
            states = torch.from_numpy(states).cuda().to(torch.float32)
        return 2 * (self.actiondecieison(states) - 0.5)
