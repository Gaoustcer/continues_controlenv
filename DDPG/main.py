from model.action import actionnet
from model.value import valuenet
from Agent.DDPG_agent import DDPG
from replaybuffer.replay import buffer


if __name__ == "__main__":
    a_net = actionnet().cuda()
    target_a_net = actionnet().cuda()
    v_net = valuenet().cuda()
    target_v_net = valuenet().cuda()
    ReplayBuffer = buffer()
    import gym
    import d4rl
    ENV = 'maze2d-umaze-v1'
    Agent = DDPG(valuenet=v_net,actionnet=a_net,targetvaluenet=target_v_net,targetactionnet=target_a_net,make_env=lambda:gym.make(ENV),rep_buffer=ReplayBuffer,logdir='../log/DDPG')
    Agent.train()