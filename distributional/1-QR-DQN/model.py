import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import num_support, batch_size, gamma

class QRDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QRDQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.num_support = num_support

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs * num_support)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        theta = x.view(-1, self.num_outputs, self.num_support)
    
        return theta
    
    def get_action(self, state):
        theta = self.forward(state)
        Q = theta.mean(dim=2, keepdim=True)
        action = torch.argmax(Q)
        return action.item()

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).long()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        theta = online_net(states)
        action = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, num_support)
        theta_a = theta.gather(1, action).squeeze(1)

        next_theta = target_net(next_states) # batch_size * action * num_support
        next_action = next_theta.mean(dim=2).max(1)[1] # batch_size
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_support)
        next_theta_a = next_theta.gather(1, next_action).squeeze(1) # batch_size * num_support

        T_theta = rewards.unsqueeze(1) + gamma * next_theta_a * masks.unsqueeze(1)

        T_theta_tile = T_theta.view(-1, num_support, 1).expand(-1, num_support, num_support)
        theta_a_tile = theta_a.view(-1, 1, num_support).expand(-1, num_support, num_support)
        
        error_loss = T_theta_tile - theta_a_tile            
        huber_loss = nn.SmoothL1Loss(reduction='none')(T_theta_tile, theta_a_tile)
        tau = torch.arange(0.5 * (1 / num_support), 1, 1 / num_support).view(1, num_support)
        
        loss = (tau - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.mean(dim=2).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss