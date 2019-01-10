import torch
import torch.nn as nn
import torch.nn.functional as F

from config import gamma, device, batch_size, sequence_length, burn_in_length

class R2D2(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(R2D2, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 128)
        self.fc_adv = nn.Linear(128, num_outputs)
        self.fc_val = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x, hidden=None):
        # x [batch_size, sequence_length, num_inputs]
        batch_size = x.size()[0]
        sequence_length = x.size()[1]
        if hidden is not None:
            out, hidden = self.lstm(x, hidden)
        else:
            out, hidden = self.lstm(x)

        out = F.relu(self.fc(out))
        adv = self.fc_adv(out)
        adv = adv.view(batch_size, sequence_length, self.num_outputs)
        val = self.fc_val(out)
        val = val.view(batch_size, sequence_length, 1)

        qvalue = val + (adv - adv.mean(dim=2, keepdim=True))

        return qvalue, hidden

    @classmethod
    def get_td_error(cls, online_net, target_net, batch, lengths):
        def slice_burn_in(item):
            return item[:, burn_in_length:, :]
        batch_size = torch.stack(batch.state).size()[0]
        states = torch.stack(batch.state).view(batch_size, sequence_length, online_net.num_inputs)
        next_states = torch.stack(batch.next_state).view(batch_size, sequence_length, online_net.num_inputs)
        actions = torch.stack(batch.action).view(batch_size, sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(batch_size, sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, sequence_length, -1)
        steps = torch.stack(batch.step).view(batch_size, sequence_length, -1)
        rnn_state = torch.stack(batch.rnn_state).view(batch_size, sequence_length, 2, -1)

        [h0, c0] = rnn_state[:, 0, :, :].transpose(0, 1)
        h0 = h0.unsqueeze(0).detach()
        c0 = c0.unsqueeze(0).detach()

        [h1, c1] = rnn_state[:, 1, :, :].transpose(0, 1)
        h1 = h1.unsqueeze(0).detach()
        c1 = c1.unsqueeze(0).detach()

        pred, _ = online_net(states, (h0, c0))
        next_pred, _ = target_net(next_states, (h1, c1))

        next_pred_online, _ = online_net(next_states, (h1, c1))

        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        steps = slice_burn_in(steps)
        next_pred_online = slice_burn_in(next_pred_online)
        
        pred = pred.gather(2, actions)

        _, next_pred_online_action = next_pred_online.max(2)
        
        target = rewards + masks * pow(gamma, steps) * next_pred.gather(2, next_pred_online_action.unsqueeze(2))

        td_error = pred - target.detach()

        for idx, length in enumerate(lengths):
            td_error[idx][length-burn_in_length:][:] = 0

        return td_error
        
    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, lengths):
        td_error = cls.get_td_error(online_net, target_net, batch, lengths)

        loss = pow(td_error, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, td_error

    def get_action(self, state, hidden):
        state = state.unsqueeze(0).unsqueeze(0)

        qvalue, hidden = self.forward(state, hidden)

        _, action = torch.max(qvalue, 2)
        return action.numpy()[0][0], hidden
