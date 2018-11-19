import os
import gym
import random
import argparse
import numpy as np

import torch
from model import ActorCritic

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_action(policy, num_actions):
    policy = policy.data.numpy()[0]
    action = np.random.choice(num_actions, 1, p=policy)[0]
    return action


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = ActorCritic(num_inputs, num_actions)
    net.load_state_dict(torch.load(args.save_path + 'model.pth'))

    net.to(device)
    net.eval()
    running_score = 0
    steps = 0

    for e in range(5):
        done = False

        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            env.render()

            steps += 1
            policy, value = net(state)
            action = get_action(policy, num_actions)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.Tensor(next_state).to(device)
            next_state = next_state.unsqueeze(0)

            score += reward
            state = next_state

        print('{} episode | score: {:.2f}'.format(e, score))


if __name__=="__main__":
    main()
