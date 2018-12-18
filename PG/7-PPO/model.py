import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from memory import BatchMaker
from config import gamma, lambda_gae, epsilon_clip, ciritic_coefficient, entropy_coefficient, epoch_k, batch_size

import warnings


class PPO(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PPO, self).__init__()
        self.t = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = torch.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

    @classmethod
    def get_gae(self, values, rewards, masks):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        previous_value = 0
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            running_advantage = running_tderror + (gamma * lambda_gae) * running_advantage * masks[t]

            returns[t] = running_return
            previous_value = values.data[t]
            advantages[t] = running_advantage

        return returns, advantages

    @classmethod
    def train_model(cls, net, transitions, optimizer):
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        
        old_policies, old_values = net(states)
        old_policies = old_policies.view(-1, net.num_outputs).detach()
        returns, advantages = net.get_gae(old_values.view(-1).detach(), rewards, masks)

        batch_maker = BatchMaker(states, actions, returns, advantages, old_policies)
        for _ in range(epoch_k):
            for _ in range(len(states) // batch_size):
                states_sample, actions_sample, returns_sample, advantages_sample, old_policies_sample = batch_maker.sample()
                
                policies, values = net(states_sample)
                values = values.view(-1)
                policies = policies.view(-1, net.num_outputs)

                ratios = ((policies / old_policies_sample) * actions_sample.detach()).sum(dim=1)
                
                
                clipped_ratios = torch.clamp(ratios, min=1.0-epsilon_clip, max=1.0+epsilon_clip)

                actor_loss = -torch.min(ratios * advantages_sample,
                                        clipped_ratios * advantages_sample).sum()

                critic_loss = (returns_sample.detach() - values).pow(2).sum()

                policy_entropy = (torch.log(policies) * policies).sum(1, keepdim=True).mean()

                loss = actor_loss + ciritic_coefficient * critic_loss - entropy_coefficient * policy_entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        
        policy = policy[0].data.numpy()
        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        
        return action
