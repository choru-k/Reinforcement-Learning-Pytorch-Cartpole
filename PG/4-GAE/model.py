import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import gamma, lambda_gae

from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'value', 'return_value', 'advantage'))

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
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
    def get_gae(cls, net, memory):
        states, rewards, masks = [], [], []
        for item in memory:
            [state, next_state, action, reward, mask] = item
            states.append(state)
            rewards.append(reward)
            masks.append(mask)
        states = torch.stack(states)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        _, values = net.forward(states)

        running_returns = 0
        running_advantages = 0

        for t in reversed(range(len(rewards) - 1)):
            next_value = values.data[t+1]
            value = values.data[t]
            running_returns = rewards[t] + running_returns * gamma * masks[t]

            delta = rewards[t] + gamma * masks[t] * value - next_value
            running_advantages = delta + gamma * lambda_gae * masks[t] * running_advantages


            returns[t] = running_returns
            advantages[t] = running_advantages

        memory2 = []
        for t, item in enumerate(memory):
            [state, next_state, action, reward, mask] = item
            return_value = returns[t]
            advantage = advantages[t]
            value = values.data[t]
            memory2.append(Transition(state, next_state, action, reward, mask, value, return_value, advantage))

        return memory2

    @classmethod
    def train_model(cls, net, optimizer, memory):
        states, next_states, actions, rewards, masks, values, return_values, advantages = Transition(*zip(*memory))

        states = torch.stack(states)
        actions = torch.stack(actions)

        values = torch.Tensor(values)
        return_values = torch.Tensor(values)
        advantages = torch.Tensor(advantages)


        policy, _ = net(states)
        policy = policy.view(-1, net.num_outputs)
        policy_action = (policy * actions).sum(dim=1)

        log_policy = torch.log(policy_action)
        loss_policy = - log_policy * advantages
        loss_value = F.mse_loss(values, return_values)
        entropy = (torch.log(policy) * policy).sum(1)

        loss = loss_policy.sum() + loss_value.sum() - 0.1 * entropy.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action
