import gym
import torch
import torch.multiprocessing as mp
import numpy as np
from model import QNet
from memory import Memory

from config import env_name, async_update_step, update_target, max_episode, device, log_interval, goal_score

class Worker(mp.Process):
    def __init__(self, online_net, target_net, optimizer, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()

        self.env = gym.make(env_name)
        self.env.seed(500)

        self.name = 'w%i' % name
        self.global_ep, self.global_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.online_net, self.target_net, self.optimizer = online_net, target_net, optimizer

    def record(self, score, epsilon, loss):
        with self.global_ep.get_lock():
            self.global_ep.value += 1
        with self.global_ep_r.get_lock():
            if self.global_ep_r.value == 0.:
                self.global_ep_r.value = score
            else:
                self.global_ep_r.value = 0.99 * self.global_ep_r.value + 0.01 * score
        if self.global_ep.value % log_interval == 0:
            print('{} , {} episode | score: {:.2f}, | epsilon: {:.2f}'.format(
                self.name, self.global_ep.value, self.global_ep_r.value, epsilon))

        self.res_queue.put([self.global_ep.value, self.global_ep_r.value, loss])


    def update_target_model(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return self.env.action_space.sample()
        else:
            return self.target_net.get_action(state)

    def run(self):
        epsilon = 1.0
        steps = 0
        while self.global_ep.value < max_episode:
            if self.global_ep_r.value > goal_score:
                break
            done = False

            score = 0
            state = self.env.reset()
            state = torch.Tensor(state).to(device)
            state = state.unsqueeze(0)

            memory = Memory(async_update_step)

            while not done:
                steps += 1

                action = self.get_action(state, epsilon)
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

                epsilon -= 0.00001
                epsilon = max(epsilon, 0.1)

                if len(memory) == async_update_step or done:
                    batch = memory.sample()
                    loss = QNet.train_model(self.online_net, self.target_net, self.optimizer, batch)
                    memory = Memory(async_update_step)
                    if done:
                        self.record(score, epsilon, loss)
                        break
                if steps % update_target == 0:
                    self.update_target_model()

            score = score if score == 500.0 else score + 1

        self.res_queue.put(None)
