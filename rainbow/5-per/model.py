import torch
import torch.nn as nn
import torch.nn.functional as F
from config import gamma

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def get_td_error(cls, online_net, target_net, state, next_state, action, reward, mask):
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        action = torch.Tensor(action)
        reward = torch.Tensor(reward)
        mask = torch.Tensor(mask)

        pred = online_net(state).squeeze(1)
        next_pred = target_net(next_state).squeeze(1)

        pred = torch.sum(pred.mul(action), dim=1)

        target = reward + mask * gamma * next_pred.max(1)[0]

        td_error = pred - target.detach()

        return td_error

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, weights):
        td_error = cls.get_td_error(online_net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask)
        loss = pow(td_error, 2) * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]
