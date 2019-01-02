import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import batch_size, gamma, quantile_embedding_dim, num_tau_sample, num_tau_prime_sample, num_quantile_sample

class IQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(IQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)
        self.phi = nn.Linear(quantile_embedding_dim, 128)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, state, tau, num_quantiles):
        input_size = state.size()[0] # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, quantile_embedding_dim)
        pi_mtx = torch.Tensor(np.pi * np.arange(0, quantile_embedding_dim)).expand(input_size * num_quantiles, quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx)

        phi = self.phi(cos_tau)
        phi = F.relu(phi)

        state_tile = state.expand(input_size, num_quantiles, self.num_inputs)
        state_tile = state_tile.flatten().view(-1, self.num_inputs)
        
        x = F.relu(self.fc1(state_tile))
        x = self.fc2(x * phi)
        z = x.view(-1, num_quantiles, self.num_outputs)

        z = z.transpose(1, 2) # [input_size, num_output, num_quantile]
        return z

    def get_action(self, state):
        tau = torch.Tensor(np.random.rand(num_quantile_sample, 1) * 0.5) # CVaR
        z = self.forward(state, tau, num_quantile_sample)
        q = z.mean(dim=2, keepdim=True)
        action = torch.argmax(q)
        return action.item()

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).long()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        z = online_net(states, tau, num_tau_sample)
        action = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, num_tau_sample)
        z_a = z.gather(1, action).squeeze(1)

        tau_prime = torch.Tensor(np.random.rand(batch_size * num_tau_prime_sample, 1))
        next_z = target_net(next_states, tau_prime, num_tau_prime_sample)
        next_action = next_z.mean(dim=2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_tau_prime_sample)
        next_z_a = next_z.gather(1, next_action).squeeze(1)

        T_z = rewards.unsqueeze(1) + gamma * next_z_a * masks.unsqueeze(1)

        T_z_tile = T_z.view(-1, num_tau_prime_sample, 1).expand(-1, num_tau_prime_sample, num_tau_sample)
        z_a_tile = z_a.view(-1, 1, num_tau_sample).expand(-1, num_tau_prime_sample, num_tau_sample)
        
        error_loss = T_z_tile - z_a_tile
        huber_loss = nn.SmoothL1Loss(reduction='none')(T_z_tile, z_a_tile)
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)
        
        loss = (tau - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.mean(dim=2).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss