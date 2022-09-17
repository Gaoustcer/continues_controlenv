import torch.nn as nn
import numpy as np
import torch
class valuenet(nn.Module):
    def __init__(self) -> None:
        super(valuenet,self).__init__()
        self.actionembeddingnet = nn.Sequential(
            nn.Linear(2,2),
            nn.ReLU(),
            nn.Linear(2,4)
        )
        self.stateembeddingnet = nn.Sequential(
            nn.Linear(2,4),
            nn.ReLU(),
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,4)
        )
        self.valuenet = nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.ReLU(),
            nn.Linear(2,1)
        )

    def forward(self,states,actions):
        if isinstance(states,np.ndarray):
            states = torch.from_numpy(states).cuda().to(torch.float32)
        if isinstance(actions,np.ndarray):
            actions = torch.from_numpy(actions).cuda().to(torch.float32)
        print(actions.shape,states.shape)
        stateembedding = self.stateembeddingnet(states)
        actionembedding = self.actionembeddingnet(actions)
        embedding = torch.concat([stateembedding,actionembedding],dim=-1)
        # print(embedding.shape,stateembedding.shape,actionembedding.shape)
        return self.valuenet(embedding)

if __name__ == "__main__":
    N = 32
    actions = np.random.random((N,2))
    states = np.random.random((N,2))
    netinst = valuenet().cuda(
    )
    ret = netinst(states,actions)
    print(ret.shape)