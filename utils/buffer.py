import numpy as np
import torch as th


class ReplayBuffer:

    def __init__(self, obs_dim, action_dim, rew_dim=1, max_size=100000):
        self.max_size = max_size
        self.ptr, self.size, = 0, 0

        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = np.array(obs).copy()
        self.next_obs[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, to_tensor=False, device=None):
        inds = np.random.choice(self.size, batch_size, replace=replace)
        experience_tuples = (self.obs[inds], self.actions[inds], self.rewards[inds], self.next_obs[inds], self.dones[inds])
        if to_tensor:
            return tuple(map(lambda x: th.tensor(x).to(device), experience_tuples))
        else:
            return experience_tuples

    def sample_obs(self, batch_size, replace=True, to_tensor=False, device=None):
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if to_tensor:
            return th.tensor(self.obs[inds]).to(device)
        else:
            return self.obs[inds]

    def get_all_data(self):
        inds = np.arange(self.size)
        return self.obs[inds], self.actions[inds], self.rewards[inds], self.next_obs[inds], self.dones[inds]

    def __len__(self):
        return self.size