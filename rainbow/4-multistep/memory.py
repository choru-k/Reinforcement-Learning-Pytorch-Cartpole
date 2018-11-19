import random
from collections import namedtuple, deque
from config import gamma, n_step
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def sample(self, batch_size):
        # 1 ~ n - step 까지의 경험 중에서 batch_size 만큼 선택
        # reward 변경
        transitions = []
        transition_start_idxes = random.sample(range(len(self.memory) - (n_step - 1)), batch_size)
        
        for transition_start_idx in transition_start_idxes:
            state = self.memory[transition_start_idx].state
            action = self.memory[transition_start_idx].action
            next_state = self.memory[transition_start_idx + n_step - 1].next_state
            mask = self.memory[transition_start_idx + n_step - 1].mask
            reward = 0
            for step in range(n_step):
                reward += (gamma ** step) * self.memory[transition_start_idx + step].reward
            transitions.append(Transition(state, next_state, action, reward, mask))
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
