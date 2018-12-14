import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import gamma, truncation_clip, delta, max_gradient_norm, trust_region_decay

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = F.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x), dim=1)
        q_value = self.fc_critic(x)
        value = (policy * q_value).sum(-1, keepdim=True).view(-1)
        return policy, q_value, value

class LocalModel(Model):
    def __init__(self, num_inputs, num_outputs):
        super(LocalModel, self).__init__(num_inputs, num_outputs)

    def pull_from_global_model(self, global_model):
        self.load_state_dict(global_model.state_dict())


    def compute_q_retraces(self, rewards, masks, values, q_actions, rho_actions, next_value):
        q_retraces = torch.zeros(rewards.size())
        q_retraces[-1] = next_value

        q_ret = q_retraces[-1]
        for step in reversed(range(len(rewards) - 1)):
            q_ret = rewards[step] + gamma * q_ret
            q_retraces[step] = q_ret
            q_ret = rho_actions[step] * (q_ret - q_actions[step]) + values[step]

        return q_retraces


    def train(self, on_policy, trajectory, average_model, global_optimizer, global_model, global_average_model):
        states, next_states, actions, rewards, masks, old_policies = trajectory
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.Tensor(actions).long().view(-1,1)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        old_policies = torch.stack(old_policies)

        states = states.view(-1, self.num_inputs)
        next_states = next_states.view(-1, self.num_inputs)
        policies, Qs, Vs = self.forward(states)

        Q_actions = Qs.gather(1, actions).view(-1)

        rhos = (policies / old_policies).clamp(max=1)
        rho_actions = rhos.gather(1, actions).view(-1)

        if masks[-1] == 0:
            Qret = 0
        else:
            Qret = Vs[-1]
        Qrets = self.compute_q_retraces(rewards, masks, Vs, Q_actions, rho_actions, Qret)
        log_policy = torch.log(policies)
        log_policy_action = log_policy.gather(1, actions).view(-1)

        actor_loss_1 = - (log_policy_action * (
            rho_actions.clamp(max=truncation_clip) * (Qrets - Vs)
        ).detach()).mean()
        actor_loss_2 = - (log_policy * (
            (1 - truncation_clip / rhos).clamp(min=0) * policies * (Qs - Vs.view(-1,1).expand_as(Qs))
        ).detach()).sum(1).mean()
        actor_loss = actor_loss_1 + actor_loss_2

        value_loss = ((Qret - Q_actions) ** 2).mean()

        g = torch.autograd.grad(-actor_loss, policies)[0]
        average_policies, _, _ = average_model(states)
        k = (average_policies / policies)

        kl = (average_policies * torch.log(average_policies / policies)).sum(1).mean(0)

        k_dot_g = (k * g).sum()
        k_dot_k = (k * k).sum()

        adj = ((k_dot_g - delta) / k_dot_k).clamp(min=0).detach()
        trust_region_actor_loss = actor_loss + adj * kl

        loss = trust_region_actor_loss + value_loss
        global_optimizer.zero_grad()

        torch.autograd.backward(policies, grad_tensors=(-g), retain_graph=True)

        value_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_gradient_norm)

        for lp, gp in zip(self.parameters(), global_model.parameters()):
            gp.grad = lp.grad
        global_optimizer.step()

        for gp, gap in zip(global_model.parameters(), global_average_model.parameters()):
            gap = trust_region_decay * gap + (1 - trust_region_decay) * gp


        # grad_actor = torch.autograd.grad(policies, self.parameters(), grad_outputs=-g, allow_unused=True,retain_graph=True)
        # grad_value = torch.autograd.grad(value_loss, self.parameters(), allow_unused=True, retain_graph=True)


        # grads = grad_actor + grad_value

        return loss

    def get_action(self, input):
        policy, _, _ = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action, policy
