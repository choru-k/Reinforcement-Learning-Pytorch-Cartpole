import gym
import torch

from model import Model
from worker import Worker
from shared_adam import SharedAdam
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp

from config import env_name, lr

def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    env.close()

    global_model = Model(num_inputs, num_actions)
    global_average_model = Model(num_inputs, num_actions)
    global_model.share_memory()
    global_average_model.share_memory()
    global_optimizer = SharedAdam(global_model.parameters(), lr=lr)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    writer = SummaryWriter('logs')

    # n = mp.cpu_count()
    n = 1
    workers = [Worker(global_model, global_average_model, global_optimizer, global_ep, global_ep_r, res_queue, i) for i in range(n)]
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
