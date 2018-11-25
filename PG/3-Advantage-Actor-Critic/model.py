import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import gamma
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
    def train_model(cls, net, optimizer, transition):
        state, next_state, action, reward, mask = transition

        policy, value = net(state)
        policy, value = policy.view(-1, net.num_outputs), value.view(-1)
        _, next_value = net(next_state)
        next_value = next_value.view(-1)

        target = reward + mask * gamma * next_value[0]
        td_error = target - value[0]

        log_policy = torch.log(policy[0])[action]
        loss_policy = - log_policy * td_error.item()
        loss_value = F.mse_loss(value[0], target.detach())
        entropy = torch.log(policy[0]) * policy[0]

        loss = loss_policy + loss_value - 0.1 * entropy.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action
