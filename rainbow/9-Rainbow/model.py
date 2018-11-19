import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from config import gamma, sigma_zero, n_step, num_support, V_max, V_min, batch_size

class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.sigma_zero = sigma_zero
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.sigma_zero / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.sigma_zero / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)


class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.dz = float(V_max - V_min) / (num_support - 1)
        self.z = torch.Tensor([V_min + i * self.dz for i in range(num_support)])


        self.fc = nn.Linear(num_inputs, 128)
        self.fc_adv = NoisyLinear(128, num_outputs * num_support)
        self.fc_val = nn.Linear(128, num_support)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc(x))
        adv = self.fc_adv(x)
        val = self.fc_val(x)

        val = val.view(-1, 1, num_support)
        adv = adv.view(-1, self.num_outputs, num_support)
        z = val + (adv - adv.mean(1, keepdim=True))
        z = z.view(-1, self.num_outputs, num_support)
        p = nn.Softmax(dim=2)(z)
        return p

    @classmethod
    def get_m(cls, _rewards, _masks, _prob_next_states_action):
        rewards = _rewards.numpy()
        masks = _masks.numpy()
        prob_next_states_action = _prob_next_states_action.detach().numpy()
        m_prob = np.zeros([batch_size, num_support], dtype=np.float32)

        dz = float(V_max - V_min) / (num_support - 1)
        batch_id = range(batch_size)
        for j in range(num_support):
            Tz = np.clip(rewards + masks * (gamma ** (n_step + 1)) * (V_min + j * dz), V_min, V_max)
            bj = (Tz - V_min) / dz

            lj = np.floor(bj).astype(np.int64)
            uj = np.ceil(bj).astype(np.int64)

            blj = (bj - lj)
            buj = (uj - bj)

            m_prob[batch_id, lj[batch_id]] += ((1 - masks) + masks * (prob_next_states_action[batch_id, j])) * buj[batch_id]
            m_prob[batch_id, uj[batch_id]] += ((1 - masks) + masks * (prob_next_states_action[batch_id, j])) * blj[batch_id]

        return m_prob

    @classmethod
    def get_loss(cls, online_net, target_net, states, next_states, actions, rewards, masks):
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.Tensor(actions).int()
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)

        z_space = online_net.z.repeat(batch_size, online_net.num_outputs, 1)
        prob_next_states_online = online_net(next_states)
        prob_next_states_target = target_net(next_states)
        Q_next_state = torch.sum(prob_next_states_online * z_space, 2)
        next_actions = torch.argmax(Q_next_state, 1)
        prob_next_states_action = torch.stack([prob_next_states_target[i, action, :] for i, action in enumerate(next_actions)])

        m_prob = cls.get_m(rewards, masks, prob_next_states_action)
        m_prob = torch.tensor(m_prob)

        m_prob = m_prob / torch.sum(m_prob, dim=1, keepdim=True)
        expand_dim_action = torch.unsqueeze(actions, -1)
        p = torch.sum(online_net(states) * expand_dim_action.float(), dim=1)
        loss = -torch.sum(m_prob * torch.log(p + 1e-20), 1)

        return loss

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, weights):
        loss = cls.get_loss(online_net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask)

        optimizer.zero_grad()
        (loss * weights).mean().backward()
        optimizer.step()

        return loss.mean()


    def get_action(self, input):
        self.reset_noise()
        p = self.forward(input)
        p = p.squeeze(0)
        z_space = self.z.repeat(self.num_outputs, 1)
        Q = torch.sum(p * z_space, dim=1)
        action = torch.argmax(Q)
        return action.item()

    def reset_noise(self):
        self.fc_adv.reset_noise()
