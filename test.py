import gym
import d4rl
import torch
env = gym.make('maze2d-umaze-v1')
done = False
state = env.reset()
reward = 0
EPOCH = 256
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log/random')
while done == False:
    next_state,r,done,_ = env.step(env.action_space.sample())
    reward += r
print("reward is",reward)
print(env.action_space)
print(env.observation_space)
print(env.reset())
print(type(env.reset()))
print(env.action_space.sample())
print(env.step(torch.rand(2)))