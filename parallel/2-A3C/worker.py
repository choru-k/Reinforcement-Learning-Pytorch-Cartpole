import gym
import torch
import torch.multiprocessing as mp
import numpy as np
from model import LocalModel
from memory import Memory
from config import env_name, n_step, max_episode, log_interval

class Worker(mp.Process):
    def __init__(self, global_model, global_optimizer, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()

        self.env = gym.make(env_name)
        self.env.seed(500)

        self.name = 'w%i' % name
        self.global_ep, self.global_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_model, self.global_optimizer = global_model, global_optimizer
        self.local_model = LocalModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.num_actions = self.env.action_space.n

    def record(self, score, loss):
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

        self.res_queue.put([self.global_ep.value, self.global_ep_r.value, loss])

    def get_action(self, policy, num_actions):
        policy = policy.data.numpy()[0]
        action = np.random.choice(num_actions, 1, p=policy)[0]
        return action

    def run(self):

        while self.global_ep.value < max_episode:
            self.local_model.pull_from_global_model(self.global_model)
            done = False
            score = 0
            steps = 0

            state = self.env.reset()
            state = torch.Tensor(state)
            state = state.unsqueeze(0)
            memory = Memory(n_step)

            while True:
                policy, value = self.local_model(state)
                action = self.get_action(policy, self.num_actions)

                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.Tensor(next_state)
                next_state = next_state.unsqueeze(0)

                mask = 0 if done else 1
                reward = reward if not done or score == 499 else -1
                action_one_hot = torch.zeros(2)
                action_one_hot[action] = 1
                memory.push(state, next_state, action_one_hot, reward, mask)

                score += reward
                state = next_state

                if len(memory) == n_step or done:
                    batch = memory.sample()
                    loss = self.local_model.push_to_global_model(batch, self.global_model, self.global_optimizer)
                    self.local_model.pull_from_global_model(self.global_model)
                    memory = Memory(n_step)

                    if done:
                        running_score = self.record(score, loss)
                        break


        self.res_queue.put(None)
