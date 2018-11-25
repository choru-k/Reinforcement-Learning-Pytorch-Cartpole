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

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = F.relu(self.fc_1(input))
        policy = F.softmax(self.fc_2(x))
        return policy

    @classmethod
    def train_model(cls, net, optimizer, transition, reward):
        state, next_state, action, _, mask = transition

        policy = net(state)
        policy = policy.view(-1, net.num_outputs)

        log_policy = torch.log(policy[0])[action]

        loss = - log_policy * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        policy = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action
