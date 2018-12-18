import random
from collections import namedtuple, deque
from config import batch_size

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))

class Memory(object):
    def __init__(self):
        self.memory = deque()

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def sample(self):
        memory = self.memory
        return Transition(*zip(*memory)) 

    def __len__(self):
        return len(self.memory)

class BatchMaker():
    def __init__(self, states, actions, returns, advantages, old_policies):
        self.states = states
        self.actions = actions
        self.returns = returns
        self.advantages = advantages
        self.old_policies = old_policies
    
    def sample(self):
        sample_indexes = random.sample(range(len(self.states)), batch_size)
        states_sample = self.states[sample_indexes]
        actions_sample = self.actions[sample_indexes]
        retruns_sample = self.returns[sample_indexes]
        advantages_sample = self.advantages[sample_indexes]
        old_policies_sample = self.old_policies[sample_indexes]
        
        return states_sample, actions_sample, retruns_sample, advantages_sample, old_policies_sample

