import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import batch_size, num_support, gamma, V_max, V_min


class Distributional_C51(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Distributional_C51, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.dz = float(V_max - V_min) / (num_support - 1)
        self.z = torch.Tensor([V_min + i * self.dz for i in range(num_support)])

        self.fc1 = nn.Linear(num_inputs, 128)
        # self.fc2 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_outputs * num_support)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)


    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        z = x.view(-1, self.num_outputs, num_support)
        p = nn.Softmax(dim=2)(z)
        return p


    def get_action(self, input):
        p = self.forward(input)
        p = p.squeeze(0)
        z_space = self.z.repeat(self.num_outputs, 1)
        Q = torch.sum(p * z_space, dim=1)
        action = torch.argmax(Q)
        return action.item()

    @classmethod
    def get_m(cls, _rewards, _masks, _prob_next_states_action):
        rewards = _rewards.numpy()
        masks = _masks.numpy()
        prob_next_states_action = _prob_next_states_action.detach().numpy()
        m_prob = np.zeros([batch_size, num_support], dtype=np.float32)

        dz = float(V_max - V_min) / (num_support - 1)
        batch_id = range(batch_size)
        for j in range(num_support):
            Tz = np.clip(rewards + masks * gamma * (V_min + j * dz), V_min, V_max)
            bj = (Tz - V_min) / dz

            lj = np.floor(bj).astype(np.int64)
            uj = np.ceil(bj).astype(np.int64)

            blj = (bj - lj)
            buj = (uj - bj)

            m_prob[batch_id, lj[batch_id]] += ((1 - masks) + masks * (prob_next_states_action[batch_id, j])) * buj[batch_id]
            m_prob[batch_id, uj[batch_id]] += ((1 - masks) + masks * (prob_next_states_action[batch_id, j])) * blj[batch_id]

        return m_prob


    @classmethod
    def train_model(cls, oneline_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).int()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        z_space = oneline_net.z.repeat(batch_size, oneline_net.num_outputs, 1)
        prob_next_states = target_net(next_states)
        Q_next_state = torch.sum(prob_next_states * z_space, 2)
        next_actions = torch.argmax(Q_next_state, 1)
        prob_next_states_action = torch.stack([prob_next_states[i, action, :] for i, action in enumerate(next_actions)])

        m_prob = cls.get_m(rewards, masks, prob_next_states_action)
        m_prob = torch.tensor(m_prob)

        m_prob = m_prob / torch.sum(m_prob, dim=1, keepdim=True)
        expand_dim_action = torch.unsqueeze(actions, -1)
        p = torch.sum(oneline_net(states) * expand_dim_action.float(), dim=1)
        loss = -torch.sum(m_prob * torch.log(p + 1e-20))
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
