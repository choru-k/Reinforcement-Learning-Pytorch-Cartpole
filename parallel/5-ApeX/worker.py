import gym
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np
from model import LocalModel
from memory import LocalBuffer, N_Step_Buffer
from config import env_name, log_interval, max_episode, n_step, local_mini_batch, gamma, batch_size, lr
from tensorboardX import SummaryWriter

class Actor(mp.Process):
    def __init__(self, global_target_model, global_memory_pipe, global_ep, global_ep_r, epsilon, name):
        super(Actor, self).__init__()

        self.env = gym.make(env_name)
        self.env.seed(500)

        self.name = 'w%i' % name
        self.global_target_model, self.global_memory_pipe = global_target_model, global_memory_pipe
        self.global_ep, self.global_ep_r = global_ep, global_ep_r
        self.local_model = LocalModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.num_actions = self.env.action_space.n

        self.local_buffer = LocalBuffer()
        self.epsilon = epsilon

    def record(self, score):
        with self.global_ep.get_lock():
            self.global_ep.value += 1
        with self.global_ep_r.get_lock():
            if self.global_ep_r.value == 0.:
                self.global_ep_r.value = score
            else:
                self.global_ep_r.value = 0.99 * self.global_ep_r.value + 0.01 * score
        if self.global_ep.value % log_interval == 0:
            print('{} , {} episode | score: {:.2f}'.format(
                self.name, self.global_ep.value, self.global_ep_r.value))

        # self.res_queue.put([self.global_ep.value, self.global_ep_r.value])

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.local_model.get_action(state)

    def run(self):
        steps = 0
        while self.global_ep.value < max_episode:
            self.local_model.pull_from_global_model(self.global_target_model) 
            done = False

            score = 0
            state = self.env.reset()
            state = torch.Tensor(state)
            state = state.unsqueeze(0)

            memory = N_Step_Buffer()

            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                next_state = torch.Tensor(next_state)
                next_state = next_state.unsqueeze(0)

                mask = 0 if done else 1
                reward = reward if not done or score == 499 else -1
                action_one_hot = np.zeros(2)
                action_one_hot[action] = 1
                memory.push(state, next_state, action_one_hot, reward, mask)
                
                score += reward
                state = next_state

                if len(memory) == n_step or done:
                    steps += 1
                    [memory_state, memory_next_state, memory_action, memory_reward, memory_mask, memory_step] = memory.sample()
                    self.local_buffer.push(memory_state, memory_next_state, memory_action, memory_reward, memory_mask, memory_step)
                    
                    if len(self.local_buffer) >= local_mini_batch:
                        # Local Buffer Get
                        transitions = self.local_buffer.sample()
                        # Copmpute Priorities
                        priors = self.compute_prior(transitions)
                        
                        # Global Memory Add
                        self.add_global_memory(transitions, priors)
                        self.local_buffer.reset()
                        self.local_model.pull_from_global_model(self.global_target_model) 
                            
            self.record(score)

    def compute_prior(self, transitions):
        states, next_states, actions, rewards, masks, steps = transitions
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        steps = torch.Tensor(steps)

        pred = self.local_model(states).squeeze(1)
        next_pred = self.local_model(next_states).squeeze(1)

        pred_action = (pred * actions).sum(dim=1)

        target = rewards + masks * pow(gamma, steps) * next_pred.max(1)[0]

        td_error = pred_action - target
        prior = abs(td_error.detach())

        return prior

    def add_global_memory(self, transitions, priors):
        self.global_memory_pipe.put([transitions, priors])
        

class Learner(mp.Process):
    def __init__(self, global_online_model, global_target_model, global_memory, global_memory_pipe, res_queue):
        super(Learner, self).__init__()
        self.online_model = global_online_model
        self.target_model = global_target_model
        self.memory = global_memory
        self.memory_pipe = global_memory_pipe
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=lr)
        self.res_queue = res_queue

    def run(self):
        train_count = 0

        while True:
            self.get_transitions()
            if len(self.memory) > batch_size:
                indexes, transitions, weights = self.memory.sample()
                loss = self.train(transitions, weights)
                self.res_queue.put([train_count, loss])
                train_count += 1
                priors = self.compute_prior(transitions)
                self.memory.update_prior(indexes, priors)

                if train_count % 30:
                    self.target_model.load_state_dict(self.online_model.state_dict())
    
    def get_transitions(self):
        while self.memory_pipe.empty() is False:
            [transitions, priors] = self.memory_pipe.get()
            states, next_states, actions, rewards, masks, steps = transitions
            for t in range(len(priors)):
                self.memory.push(states[t], next_states[t], actions[t], rewards[t], masks[t], steps[t], priors[t])
                

    def get_td_error(self, transitions):
        states, next_states, actions, rewards, masks, steps = transitions

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        steps = torch.Tensor(steps)
        
        pred = self.online_model(states).squeeze(1)
        _, action_from_online_net = self.online_model(next_states).squeeze(1).max(1)
        next_pred = self.target_model(next_states).squeeze(1)

        pred_action = (pred * actions).sum(dim=1)

        target = rewards + masks * pow(gamma, steps) * next_pred.gather(1, action_from_online_net.unsqueeze(1)).squeeze(1)

        td_error = pred_action - target.detach()

        return td_error

    def train(self, transitions, weights):
        td_error = self.get_td_error(transitions)
        loss = (pow(td_error, 2) * weights).mean()
        # loss = (pow(td_error, 2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def compute_prior(self, transitions):
        td_error = self.get_td_error(transitions)

        prior = abs(td_error.detach())

        return prior

