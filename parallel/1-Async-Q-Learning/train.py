import os
import sys
import gym
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from worker import Worker
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
from shared_adam import SharedAdam

from config import env_name, lr, device


def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    target_net.load_state_dict(online_net.state_dict())
    online_net.share_memory()
    target_net.share_memory()

    optimizer = SharedAdam(online_net.parameters(), lr=lr)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()

    workers = [Worker(online_net, target_net, optimizer, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            [ep, ep_r, loss] = r
            writer.add_scalar('log/score', float(ep_r), ep)
            writer.add_scalar('log/loss', float(loss), ep)
        else:
            break
    [w.join() for w in workers]


if __name__=="__main__":
    main()
