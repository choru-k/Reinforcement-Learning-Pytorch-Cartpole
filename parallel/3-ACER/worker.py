import gym
import torch
import torch.multiprocessing as mp
import numpy as np
from model import LocalModel
from memory import Memory, Trajectory
from config import env_name, max_episode, log_interval, replay_memory_capacity, replay_ratio

class Worker(mp.Process):
    def __init__(self, global_model, global_average_model, global_optimizer, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()

        self.env = gym.make(env_name)
        self.env.seed(500)

        self.name = 'w%i' % name
        self.global_ep, self.global_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_model, self.global_average_model, self.global_optimizer = global_model, global_average_model, global_optimizer
        self.local_model = LocalModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.num_actions = self.env.action_space.n

        self.memory = Memory(replay_memory_capacity)

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

    def run(self):
        while self.global_ep.value < max_episode:
            self.algorithm(True)
            n = np.random.poisson(replay_ratio)
            for _ in range(n):
                self.algorithm(False)

    def algorithm(self, on_policy):
        self.local_model.pull_from_global_model(self.global_model)
        if not on_policy and len(self.memory) > 100:
            trajectory = self.memory.sample()
        else:
            trajectory, score = self.run_env()
        loss = self.local_model.train(on_policy, trajectory, self.global_average_model, self.global_optimizer, self.global_model, self.global_average_model)
        if on_policy:
            self.record(score, loss)


    def run_env(self):
        done = False
        score = 0
        steps = 0

        state = self.env.reset()
        state = torch.Tensor(state)
        state = state.unsqueeze(0)
        trajectory = Trajectory()

        while True:
            action, policy = self.local_model.get_action(state)
            policy = torch.Tensor(policy)

            next_state, reward, done, _ = self.env.step(action)
            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1
            trajectory.push(state, next_state, action, reward, mask, policy)

            score += reward
            state = next_state

            if done:
                break

        self.memory.push(trajectory)
        trajectory = trajectory.sample()
        return trajectory, score
