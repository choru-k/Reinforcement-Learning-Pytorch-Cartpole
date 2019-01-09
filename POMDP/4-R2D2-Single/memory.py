import random
from collections import namedtuple, deque
from config import sequence_length, burn_in_length, eta
import torch
import numpy as np

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'rnn_state'))


class LocalBuffer(object):
    def __init__(self):
        self.local_memory = []
        self.memory = []
    
    def push(self, state, next_state, action, reward, mask, rnn_state):
        self.local_memory.append(Transition(state, next_state, action, reward, mask, torch.stack(rnn_state).view(2, -1)))
        if len(self.local_memory) == sequence_length or mask == 0:
            if len(self.local_memory) > burn_in_length:
                length = len(self.local_memory)
                while len(self.local_memory) < sequence_length:
                    self.local_memory.append(Transition(
                        torch.Tensor([0, 0]),
                        torch.Tensor([0, 0]),
                        0,
                        0,
                        0,
                        torch.zeros([2, 1, 16]).view(2, -1)
                    )) 
                self.memory.append([self.local_memory, length])
            self.local_memory = []
    
    def sample(self):
        episodes = self.memory
        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_rnn_state = [], [], [], [], [], []
        lengths = []
        for episode, length in episodes:
            batch = Transition(*zip(*episode))

            batch_state.append(torch.stack(list(batch.state)))
            batch_next_state.append(torch.stack(list(batch.next_state)))
            batch_action.append(torch.Tensor(list(batch.action)))
            batch_reward.append(torch.Tensor(list(batch.reward)))
            batch_mask.append(torch.Tensor(list(batch.mask)))
            batch_rnn_state.append(torch.stack(list(batch.rnn_state)))

            lengths.append(length)
        self.memory = []
        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_rnn_state), lengths



class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.memory_probability = deque(maxlen=capacity)

    def td_error_to_prior(self, td_error, lengths):
        abs_td_error_sum  = td_error.sum(dim=1, keepdim=True).view(-1).abs().detach().numpy()
        lengths_burn = [length - burn_in_length for length in lengths]
        
        prior = abs_td_error_sum / lengths_burn
        return prior 

    def push(self, td_error, batch, lengths):
        # batch.state[local_mini_batch, sequence_length, item]
        prior = self.td_error_to_prior(td_error, lengths)

        for i in range(len(batch)):
            if len(self.memory_probability) > 0:
                memory_probability = np.array(self.memory_probability)
                probability_max = max(memory_probability.max(), prior[i])
                probability_mean = (memory_probability.sum() + prior[i]) / (len(self.memory_probability) + 1)
            else:
                probability_max = prior[i]
                probability_mean = prior[i]
            self.memory.append([Transition(batch.state[i], batch.next_state[i], batch.action[i], batch.reward[i], batch.mask[i], batch.rnn_state[i]), lengths[i]])
            p = eta * probability_max + (1 - eta) * probability_mean
            self.memory_probability.append(p)
        
    def sample(self, batch_size):
        probability = np.array(self.memory_probability)
        probability = probability / probability.sum()

        indexes = np.random.choice(range(len(self.memory_probability)), batch_size, p=probability)
        # indexes = np.random.choice(range(len(self.memory_probability)), batch_size)
        episodes = [self.memory[idx][0] for idx in indexes]
        lengths = [self.memory[idx][1] for idx in indexes]

        batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_rnn_state = [], [], [], [], [], []
        for episode in episodes:
            batch_state.append(episode.state)
            batch_next_state.append(episode.next_state)
            batch_action.append(episode.action)
            batch_reward.append(episode.reward)
            batch_mask.append(episode.mask)
            batch_rnn_state.append(episode.rnn_state)
        
        return Transition(batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_rnn_state), indexes, lengths
    
    def update_prior(self, indexes, td_error, lengths):
        prior = self.td_error_to_prior(td_error, lengths)
        priors_idx = 0
        for idx in indexes:
            self.memory_probability[idx] = prior[priors_idx]
            priors_idx += 1
    
    def __len__(self):
        return len(self.memory)
