import d4rl
import gym
env = gym.make("maze2d-umaze-v1")
dataset = env.get_dataset()
dataset = d4rl.qlearning_dataset(env)
print(dataset.keys())
# print(dataset['observations'][:1])
# print(dataset.keys())
# print(dataset['terminals'])