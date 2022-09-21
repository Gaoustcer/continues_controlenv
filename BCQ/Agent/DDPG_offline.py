import torch.nn as nn
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
import d4rl
class DDPG(object):
    def __init__(self,valuenet,actionnet,targetvaluenet,targetactionnet,make_env,logdir) -> None:
        self.valuenet = valuenet
        # self.replay_buffer = rep_buffer
        self.targetvaluenet = targetvaluenet
        self.actionnet = actionnet
        self.targetactionnet = targetactionnet
        # self.trainenv = make_env()
        self.testenv = make_env()
        self.static_dataset = d4rl.qlearning_dataset(self.testenv)
        self.datasetlen = self.static_dataset['observations'].shape[0]
        self.NOISE = 0
        self.writer = SummaryWriter(logdir+str(self.NOISE))
        self.tau = 0.1 # soft update parameters
        self.gamma = 0.98 # discount factor
        self.EPOCH = 256 # total train epoch
        self.actionvaluelossfunction = nn.MSELoss()
        self.actionvalueoptim = torch.optim.Adam(self.valuenet.parameters(),lr = 0.0001)
        self.actionoptim = torch.optim.Adam(self.actionnet.parameters(),lr=0.0001)
        self.lossindex = 1
        self.valueindex = 1
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
                action = self.actionnet(state).cpu().detach().numpy()
                state,r,done,_ = self.testenv.step(action)
                reward += r
        return reward/K
    
    def randomselect(self,sample_size):
        from random import choices
        sampleindex = choices(range(self.datasetlen),k=sample_size)
        return self.static_dataset['observations'][sampleindex],self.static_dataset['actions'][sampleindex],self.static_dataset['rewards'][sampleindex],self.static_dataset['next_observations'][sampleindex]

        pass
    # def collectdataforkeposide(self,K=4,noise_times = 1):
    #     for epoch in range(K):
    #         done = False
    #         state = self.trainenv.reset()
    #         while done == False:
    #             action = self.actionnet(state).cpu().detach().numpy()
    #             # print("action is",action)
    #             # NOISE = 0.1
    #             action = np.clip(np.random.normal(action,self.NOISE/noise_times),-1,1)
    #             # exit()
    #             ns,r,done,_ = self.trainenv.step(action)
    #             self.replay_buffer.push_memory(state,action,r,ns)
    #             state = ns
    
    def updateparameters(self,sample_size = 64,actionvalueupdatetime=16,actionupdatetime=4):
        currentstate,action,reward,nextstate = self.randomselect(sample_size)
        # currentstate,action,reward,nextstate = self.replay_buffer.sample(sample_size)
        '''
        update action_value net based on TDloss
        '''
        # print("action is",action)
        # exit()
        for _ in range(actionvalueupdatetime):
            currentactionvalue = self.valuenet(currentstate,action).squeeze()
            nextactionvalue = (self.gamma * self.targetvaluenet(nextstate,self.targetactionnet(nextstate)).squeeze() + torch.from_numpy(reward).cuda().to(torch.float32)).detach()
            # print("current value is",currentactionvalue)
            # print("self.targetvalue",self.targetvaluenet(nextstate,self.targetactionnet(nextstate)).squeeze())
            # print("next action is ",nextactionvalue)
            # print("judge is",torch.from_numpy(reward))
            # exit()
            loss = self.actionvaluelossfunction(currentactionvalue,nextactionvalue)
            self.actionvalueoptim.zero_grad()
            loss.backward()
            self.writer.add_scalar('TDloss',loss,self.lossindex)
            self.lossindex += 1
            self.actionvalueoptim.step()
        for _ in range(actionupdatetime):
            values = -torch.mean(self.valuenet(currentstate,self.actionnet(currentstate)))
            self.actionoptim.zero_grad()
            # print("values is",values)
            # exit()
            values.backward()
            self.writer.add_scalar('value',-values,self.valueindex)
            self.valueindex += 1
            self.actionoptim.step()
    
    def _softupdate(self):
        self._soft_update(self.actionnet,self.targetactionnet)
        self._soft_update(self.valuenet,self.targetvaluenet)
    
    def _random(self):
        from tqdm import tqdm
        self.baselinewriter = SummaryWriter("../log/mazebaseline")
        for epoch in tqdm(range(self.EPOCH//16)):
            reward = self.validation()
            self.baselinewriter.add_scalar('reward',reward,epoch)

    
    def train(self,sample_time=64):
        # while self.replay_buffer.full == False:
        #     self.collectdataforkeposide()
        from tqdm import tqdm
        for epoch in tqdm(range(self.EPOCH)):
            # self.collectdataforkeposide(K=2,noise_times = epoch//16+1)
            for _ in range(sample_time):
                self.updateparameters()
            reward = self.validation()
            self.writer.add_scalar('reward',reward,epoch)
            self._softupdate()
    
    
        

