import torch
import torch.nn as nn
import torch.nn.functional as F

from config import gamma, device, batch_size, sequence_length, burn_in_length

class DRQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DRQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x, hidden=None):
        # x [batch_size, sequence_length, num_inputs]

        if hidden is not None:
            out, hidden = self.lstm(x, hidden)
        else:
            out, hidden = self.lstm(x)
        out = F.relu(self.fc1(out))
        qvalue = self.fc2(out)

        return qvalue, hidden


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        def slice_burn_in(item):
            return item[:, burn_in_length:, :]
        states = torch.stack(batch.state).view(batch_size, sequence_length, online_net.num_inputs)
        next_states = torch.stack(batch.next_state).view(batch_size, sequence_length, online_net.num_inputs)
        actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)

        pred, _ = online_net(states)
        next_pred, _ = target_net(next_states)

        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        
        pred = pred.gather(2, actions)
        
        target = rewards + masks * gamma * next_pred.max(2, keepdim=True)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, state, hidden):
        state = state.unsqueeze(0).unsqueeze(0)

        qvalue, hidden = self.forward(state, hidden)

        _, action = torch.max(qvalue, 2)
        
        return action.numpy()[0][0], hidden
