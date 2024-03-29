import numpy as np
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device = None):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.memory_pointer, self.size, self.max_size = 0, 0, size
        if device == None:
            self.device = 'cpu'
        else:
            self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.memory_pointer] = obs
        self.obs2_buf[self.memory_pointer] = next_obs
        self.act_buf[self.memory_pointer] = act
        self.rew_buf[self.memory_pointer] = rew
        self.done_buf[self.memory_pointer] = done
        self.memory_pointer = (self.memory_pointer+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k,v in batch.items()}