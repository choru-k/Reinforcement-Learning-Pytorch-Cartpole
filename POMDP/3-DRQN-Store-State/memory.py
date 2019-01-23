import random
from collections import namedtuple, deque
from config import sequence_length
import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'rnn_state'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.local_memory = []
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask, rnn_state):
        self.local_memory.append(Transition(state, next_state, action, reward, mask, torch.stack(rnn_state).view(2, -1)))
        if mask == 0:
            self.memory.append(self.local_memory)
            self.local_memory = []

    def sample(self, batch_size):
        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_rnn_state = [], [], [], [], [], []
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
            batch_rnn_state.append(torch.stack(list(batch.rnn_state)))
        
        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_rnn_state)

    def __len__(self):
        return len(self.memory)