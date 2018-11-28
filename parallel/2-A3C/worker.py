import gym
import torch
import torch.multiprocessing as mp
import numpy as np
from model import LocalModel
from memory import Memory

def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r:", global_ep_r.value,
    )

class Worker(mp.Process):
    def __init__(self, global_model, global_optimizer, global_ep, global_ep_r, res_queue, name, args):
        super(Worker, self).__init__()
        self.args = args

        self.env = gym.make(self.args.env_name)
        self.env.seed(500)

        self.name = 'w%i' % name
        self.global_ep, self.global_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_model, self.global_optimizer = global_model, global_optimizer
        self.local_model = LocalModel(self.env.observation_space.shape[0], self.env.action_space.n)
        self.num_actions = self.env.action_space.n


    def get_action(self, policy, num_actions):
        policy = policy.data.numpy()[0]
        action = np.random.choice(num_actions, 1, p=policy)[0]
        return action

    def run(self):
        self.local_model.train()
        total_step = 1
        while self.global_ep.value < self.args.MAX_EP:
            self.local_model.pull_from_global_model(self.global_model)
            done = False
            score = 0
            steps = 0

            state = self.env.reset()
            state = torch.Tensor(state)
            state = state.unsqueeze(0)
            memory = Memory(100)

            while True:
                self.local_model.eval()
                policy, value = self.local_model(state)
                action = self.get_action(policy, self.num_actions)

                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.Tensor(next_state)
                next_state = next_state.unsqueeze(0)

                mask = 0 if done else 1
                reward = reward if not done or score == 499 else -1
                score += reward

                memory.push(state, next_state, action, reward, mask)

                if len(memory) == 10 or done:
                    batch = memory.sample()
                    self.local_model.push_to_global_model(batch, self.global_model, self.global_optimizer, self.args)
                    self.local_model.pull_from_global_model(self.global_model)
                    memory = Memory(100)

                    if done:
                        record(self.global_ep, self.global_ep_r, score, self.res_queue, self.name)
                        break


                total_step += 1
                state = next_state
        self.res_queue.put(None)
