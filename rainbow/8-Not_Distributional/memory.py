import random
import numpy as np
from collections import namedtuple, deque
import torch
from model import QNet
from config import small_epsilon, gamma, alpha, device, n_step

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class Memory_With_TDError(object):
    def __init__(self, capacity):
        self.memory = []
        self.memory_probabiliy = []
        self.capacity = capacity
        self.position = 0
        self.reset_local()

    def reset_local(self):
        self.local_step = 0
        self.local_state = None
        self.local_action = None
        self.local_rewards = []

    def push(self, state, next_state, action, reward, mask):
        self.local_step += 1
        self.local_rewards.append(reward)
        if self.local_step == 1:
            self.local_state = state
            self.local_action = action
        if self.local_step == n_step:
            reward = 0
            for idx, local_reward in enumerate(self.local_rewards):
                reward += (gamma ** idx) * local_reward
            self.push_to_memory(self.local_state, next_state, self.local_action, reward, mask)
            self.reset_local()
        if mask == 0:
            self.reset_local()


    def push_to_memory(self, state, next_state, action, reward, mask):
        if len(self.memory) > 0:
            max_probability = max(self.memory_probabiliy)
        else:
            max_probability = small_epsilon

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, next_state, action, reward, mask))
            self.memory_probabiliy.append(max_probability)
        else:
            self.memory[self.position] = Transition(state, next_state, action, reward, mask)
            self.memory_probabiliy[self.position] = max_probability

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, net, target_net, beta):
        probability_sum = sum(self.memory_probabiliy)
        p = [probability / probability_sum for probability in self.memory_probabiliy]

        indexes = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)
        transitions = [self.memory[idx] for idx in indexes]
        transitions_p = [p[idx] for idx in indexes]
        batch = Transition(*zip(*transitions))

        weights = [pow(self.capacity * p_j, -beta) for p_j in transitions_p]
        weights = torch.Tensor(weights).to(device)
        weights = weights / weights.max()


        td_error = QNet.get_td_error(net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask)
        td_error = td_error.detach()

        td_error_idx = 0
        for idx in indexes:
            self.memory_probabiliy[idx] = pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item()
            # print(pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item())
            td_error_idx += 1


        return batch, weights

    def __len__(self):
        return len(self.memory)
