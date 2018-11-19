import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import gamma, sigma_zero, n_step

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

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_adv = NoisyLinear(128, num_outputs)
        self.fc_val = nn.Linear(128, 1)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc(x))
        adv = self.fc_adv(x)
        val = self.fc_val(x)

        qvalue = val + (adv - adv.mean())
        return qvalue

    @classmethod
    def get_td_error(cls, oneline_net, target_net, states, next_states, actions, rewards, masks):
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.Tensor(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)

        pred = oneline_net(states).squeeze(1)
        _, action_from_oneline_net = oneline_net(next_states).squeeze(1).max(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.gather(1, action_from_oneline_net.unsqueeze(1)).squeeze(1)

        td_error = pred - target.detach()

        return td_error

    @classmethod
    def train_model(cls, oneline_net, target_net, optimizer, batch, weights):
        td_error = cls.get_td_error(oneline_net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask)

        loss = pow(td_error, 2) * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def get_action(self, input):
        self.fc_adv.reset_noise()
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]
