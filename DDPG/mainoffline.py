from model.action import actionnet
from model.value import valuenet
# from Agent.DDPG_agent import DDPG
# from replaybuffer.replay import buffer
from Agent.DDPG_offline import DDPG
from env.contin_cartpole import ContinuousCartPoleEnv
if __name__ == "__main__":
    a_net = actionnet(state_size=4,action_size=2).cuda()
    target_a_net = actionnet(state_size=4,action_size=2).cuda()
    v_net = valuenet(state_size=4,action_size=2).cuda()
    target_v_net = valuenet(state_size=4,action_size=2).cuda()
    # ReplayBuffer = buffer()
    import gym
    import d4rl
    ENV = 'maze2d-umaze-v1'
    
    Agent = DDPG(valuenet=v_net,actionnet=a_net,targetvaluenet=target_v_net,targetactionnet=target_a_net,make_env=lambda:gym.make(ENV),logdir='../log/DDPGmazeoffline')
    # Agent.train()
    Agent._random()
    Agent.train()