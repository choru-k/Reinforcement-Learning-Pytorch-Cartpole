import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import gamma, lambda_gae, ciritic_coefficient, entropy_coefficient

from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'value', 'return_value', 'advantage'))

class GAE(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(GAE, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = F.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x))
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

        policies, values = net(states)
        policies = policies.view(-1, net.num_outputs)
        values = values.view(-1)

        returns, advantages = net.get_gae(values.view(-1).detach(), rewards, masks)

        log_policies = (torch.log(policies) * actions.detach()).sum(dim=1)
        actor_loss = -(log_policies * advantages).sum()
        critic_loss = (returns.detach() - values).pow(2).sum()
        
        entropy = (torch.log(policies) * policies).sum(1).sum()

        loss = actor_loss + ciritic_coefficient * critic_loss - entropy_coefficient * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action
