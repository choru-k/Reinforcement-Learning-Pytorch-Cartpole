import os
import sys
import gym
import argparse
import numpy as np

from memory import Memory
from model import Model
from worker import Worker
from shared_adam import SharedAdam
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--gamma', default=0.9, help='')
parser.add_argument('--goal_score', default=400, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
parser.add_argument('--MAX_EP', default=10000)
args = parser.parse_args()

if __name__ == "__main__":
    env = gym.make(args.env_name)
    global_model = Model(env.observation_space.shape[0], env.action_space.n)
    global_model.share_memory()
    global_optimizer = SharedAdam(global_model.parameters(), lr=0.0001)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # mp.cpu_count()
    workers = [Worker(global_model, global_optimizer, global_ep, global_ep_r, res_queue, i, args) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    # import matplotlib.pyplot as plt
    # plt.plot(res)
    # plt.ylabel('Moving average ep reward')
    # plt.xlabel('Step')
    # plt.show()
