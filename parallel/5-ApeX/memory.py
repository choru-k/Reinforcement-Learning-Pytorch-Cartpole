import random
import numpy as np
import torch
from collections import namedtuple, deque

from config import gamma, batch_size, alpha, beta

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'step'))

class N_Step_Buffer(object):
    def __init__(self): 
        self.memory = []
        self.step = 0

    def push(self, state, next_state, action, reward, mask):
        self.step += 1
        self.memory.append([state, next_state, action, reward, mask])        

    def sample(self):
        [state, _, action, _, _] = self.memory[0]
        [_, next_state, _, _, mask] = self.memory[-1]

        sum_reward = 0
        for t in reversed(range(len(self.memory))):
            [_, _, _, reward, _] = self.memory[t]
            sum_reward += reward + gamma * sum_reward
        reward = sum_reward
        step = self.step
        self.reset()

        return [state, next_state, action, reward, mask, step]

    def reset(self):
        self.memory = []
        self.step = 0
    
    def __len__(self):
        return len(self.memory)


class LocalBuffer(object):
    def __init__(self):
        self.memory = []
    
    def push(self, state, next_state, action, reward, mask, step):
        self.memory.append(Transition(state, next_state, action, reward, mask, step))
    
    def sample(self):
        transitions = self.memory
        batch = Transition(*zip(*transitions))
        return batch
    
    def reset(self):
        self.memory = []
    
    def __len__(self):
        return len(self.memory)

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.memory_probability = deque(maxlen=capacity)
    
    def push(self, state, next_state, action, reward, mask, step, prior):
        self.memory.append(Transition(state, next_state, action, reward, mask, step))
        self.memory_probability.append(prior)

    def sample(self):
        probaility = torch.Tensor(self.memory_probability)
        probaility = probaility.pow(alpha)
        probaility = probaility / probaility.sum()

        p = probaility.numpy()

        indexes = np.random.choice(range(len(self.memory_probability)), batch_size, p=p)
        
        transitions = [self.memory[idx] for idx in indexes]
        transitions_p = torch.Tensor([self.memory_probability[idx] for idx in indexes])
        
        batch = Transition(*zip(*transitions))

        weights = (self.capacity * transitions_p).pow(-beta)
        weights = weights / weights.max()

        return indexes, batch, weights

    def update_prior(self, indexes, priors):
        priors_idx = 0
        for idx in indexes:
            self.memory_probability[idx] = priors[priors_idx]
            priors_idx += 1
    
    def __len__(self):
        return len(self.memory)
        
    
    