import random
from collections import namedtuple, deque
from config import sequence_length
import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.local_memory = []
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask):
        self.local_memory.append(Transition(state, next_state, action, reward, mask))
        if mask == 0:
            while len(self.local_memory) < sequence_length:
                self.local_memory.insert(0, Transition(
                    torch.Tensor([0, 0]),
                    torch.Tensor([0, 0]),
                    0,
                    0,
                    0,
                ))
            self.memory.append(self.local_memory)
            self.local_memory = []

    def sample(self, batch_size):
        batch_state, batch_next_state, batch_action, batch_reward, batch_mask = [], [], [], [], []
        p = np.array([len(episode) for episode in self.memory])
        p = p / p.sum()

        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)
        
        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]
            
            start = random.randint(0, len(episode) - sequence_length)
            transitions = episode[start:start + sequence_length]
            batch = Transition(*zip(*transitions))

            batch_state.append(torch.stack(list(batch.state)))
            batch_next_state.append(torch.stack(list(batch.next_state)))
            batch_action.append(torch.Tensor(list(batch.action)))
            batch_reward.append(torch.Tensor(list(batch.reward)))
            batch_mask.append(torch.Tensor(list(batch.mask)))
        
        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask)

    def __len__(self):
        return len(self.memory)