import gym
import torch

from model import Model
from worker import Actor, Learner
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from memory import Memory
from config import env_name, lr, replay_memory_capacity

def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    env.close()

    global_target_model = Model(num_inputs, num_actions)
    global_online_model = Model(num_inputs, num_actions)
    global_target_model.train()
    global_online_model.train()
    
    global_target_model.load_state_dict(global_online_model.state_dict())
    global_target_model.share_memory()
    global_online_model.share_memory()
    
    global_memory = Memory(replay_memory_capacity)
    
    
    global_ep, global_ep_r, res_queue, global_memory_pipe = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue(), mp.Queue()

    writer = SummaryWriter('logs')

    n = 2 
    epsilons = [(i * 0.05 + 0.1) for i in range(n)]

    actors = [Actor(global_target_model, global_memory_pipe, global_ep, global_ep_r, epsilons[i], i) for i in range(n)]
    [w.start() for w in actors]
    learner = Learner(global_online_model, global_target_model, global_memory, global_memory_pipe, res_queue)
    learner.start()

    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            [ep, loss] = r
            # writer.add_scalar('log/score', float(ep_r), ep)
            writer.add_scalar('log/loss', float(loss), ep)
        else:
            break
    [w.join() for w in actors]

if __name__=="__main__":
    main()
