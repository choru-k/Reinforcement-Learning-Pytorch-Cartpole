import random
import numpy as np
from collections import namedtuple, deque
import torch
from model import QNet
from config import small_epsilon, gamma, alpha, device

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class Memory_With_TDError(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.memory_probabiliy = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask):
        """Saves a transition."""
        if len(self.memory) > 0:
            max_probability = max(self.memory_probabiliy)
        else:
            max_probability = small_epsilon
        self.memory.append(Transition(state, next_state, action, reward, mask))
        self.memory_probabiliy.append(max_probability)

    def sample(self, batch_size, net, target_net, beta):
        probability_sum = sum(self.memory_probabiliy)
        p = [probability / probability_sum for probability in self.memory_probabiliy]
        # print(len(self.memory_probabiliy))
        indexes = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)
        transitions = [self.memory[idx] for idx in indexes]
        transitions_p = [p[idx] for idx in indexes]
        batch = Transition(*zip(*transitions))

        weights = [pow(self.capacity * p_j, -beta) for p_j in transitions_p]
        weights = torch.Tensor(weights).to(device)
        # print(weights)
        weights = weights / weights.max()
        # print(weights)

        td_error = QNet.get_td_error(net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask)

        td_error_idx = 0
        for idx in indexes:
            self.memory_probabiliy[idx] = pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item()
            # print(pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item())
            td_error_idx += 1


        return batch, weights

    def __len__(self):
        return len(self.memory)
