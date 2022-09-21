from torch.utils.data import Dataset
import gym
import d4rl
class OfflineRL(Dataset):
    def __init__(self) -> None:
        super(OfflineRL,self).__init__()
        self.env = gym.make('maze2d-umaze-v1')
        self.dataset = d4rl.qlearning_dataset(self.env)
        self.observations = self.dataset['observations']
        self.actions = self.dataset['actions']
        self.rewards = self.dataset['rewards']
        self.nextobservations = self.dataset['next_observations']
        print("Init dataset")
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, index):
        # return super().__getitem__(index)
        return self.observations[index],self.actions[index]