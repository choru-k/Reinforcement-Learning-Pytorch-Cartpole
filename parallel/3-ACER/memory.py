import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'policy'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, trajectory):
        self.memory.append(trajectory.trajectory)

    def sample(self):
        trajectory = self.memory[random.randrange(len(self.memory))]
        return Transition(*zip(*trajectory))

    def __len__(self):
        return len(self.memory)

class Trajectory(object):
    def __init__(self):
        self.trajectory = []

    def push(self, state, next_state, action, reward, mask, policy):
        self.trajectory.append(Transition(state, next_state, action, reward, mask, policy))

    def sample(self):
        trajectory = self.trajectory
        return Transition(*zip(*trajectory))

    def __len__(self):
        return len(self.trajectory)
