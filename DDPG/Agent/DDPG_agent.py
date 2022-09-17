import torch.nn as nn
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

class DDPG(object):
    def __init__(self,valuenet,actionnet,targetvaluenet,targetactionnet,make_env,rep_buffer,logdir) -> None:
        self.valuenet = valuenet
        self.replay_buffer = rep_buffer
        self.targetvaluenet = targetvaluenet
        self.actionnet = actionnet
        self.targetactionnet = targetactionnet
        self.trainenv = make_env()
        self.testenv = make_env()
        self.writer = SummaryWriter(logdir)
        self.tau = 0.2
        self.EPOCH = 256
        self.actionvaluelossfunction = nn.MSELoss()
        self.actionvalueoptim = torch.optim.Adam(self.valuenet.parameters(),lr = 0.001)
        self.actionoptim = torch.optim.Adam(self.actionnet.parameters(),lr=0.001)
    def _soft_update(self,origin_net:nn.Module,target_net:nn.Module):
        for targetparam,originparam in zip(target_net.parameters(),origin_net.parameters()):
            targetparam.data.copy_(
                self.tau * originparam + (1 - self.tau) * targetparam
            )
    def validation(self,K=16):
        reward = 0
        for epoch in range(K):
            state = self.testenv.reset()
            done = False
            while done == False:
                action = self.actionnet(state).cpu().detach()
                state,r,done,_ = self.testenv.step(action)
                reward += r
        return reward/K

    def collectdataforkeposide(self,K=4):
        for epoch in range(K):
            done = False
            state = self.trainenv.reset()
            while done == False:
                action = self.actionnet(state).cpu().detach().numpy()
                ns,r,done,_ = self.trainenv.step(action)
                self.replay_buffer.push_memory(state,action,r,ns)
                state = ns
    
    def updateparameters(self,sample_size = 64,actionvalueupdatetime=4,actionupdatetime=4):
        currentstate,action,reward,nextstate = self.replay_buffer.sample(sample_size)
        '''
        update action_value net based on TDloss
        '''
        # print("action is",action)
        # exit()
        for _ in range(actionvalueupdatetime):
            currentactionvalue = self.valuenet(currentstate,action)
            nextactionvalue = (self.targetvaluenet(nextstate,self.targetactionnet(nextstate)) + torch.from_numpy(reward).cuda().to(torch.float32)).detach()
            loss = self.actionvaluelossfunction(currentactionvalue,nextactionvalue)
            self.actionvalueoptim.zero_grad()
            loss.backward()
            self.actionvalueoptim.step()
        for _ in range(actionupdatetime):
            values = -torch.sum(self.valuenet(currentstate,self.actionnet(currentstate)))
            self.actionoptim.zero_grad()
            values.backward()
            self.actionoptim.step()
    
    def _softupdate(self):
        self._soft_update(self.actionnet,self.targetactionnet)
        self._soft_update(self.valuenet,self.targetvaluenet)
    
    def _random(self):
        from tqdm import tqdm
        for epoch in tqdm(range(self.EPOCH)):
            reward = self.validation()
            self.writer.add_scalar('reward/baseline',reward,epoch)

    
    def train(self,sample_time=8):
        while self.replay_buffer.full == False:
            self.collectdataforkeposide()
        from tqdm import tqdm
        for epoch in tqdm(range(self.EPOCH)):
            self.collectdataforkeposide(K=2)
            for _ in range(sample_time):
                self.updateparameters()
            reward = self.validation()
            self.writer.add_scalar('reward',reward,epoch)
    
    
        

