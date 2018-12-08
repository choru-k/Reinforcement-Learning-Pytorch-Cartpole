import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import gamma, max_kl

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten

def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten

def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten

def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

def kl_divergence(policy, old_policy):
    kl = old_policy * torch.log(old_policy / policy)

    kl = kl.sum(1, keepdim=True)
    return kl

def fisher_vector_product(net, states, p, cg_damp=0.1):
    policy = net(states)
    old_policy = net(states).detach()
    kl = kl_divergence(policy, old_policy)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, net.parameters(), create_graph=True) # create_graph is True if we need higher order derivative products
    kl_grad = flat_grad(kl_grad)

    kl_grad_p = (kl_grad * p.detach()).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, net.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + cg_damp * p.detach()


def conjugate_gradient(net, states, loss_grad, n_step=10, residual_tol=1e-10):
    x = torch.zeros(loss_grad.size())
    r = loss_grad.clone()
    p = loss_grad.clone()
    r_dot_r = torch.dot(r, r)

    for i in range(n_step):
        A_dot_p = fisher_vector_product(net, states, p)
        alpha = r_dot_r / torch.dot(p, A_dot_p)
        x += alpha * p
        r -= alpha * A_dot_p
        new_r_dot_r = torch.dot(r,r)
        betta = new_r_dot_r / r_dot_r
        p = r + betta * p
        r_dot_r = new_r_dot_r
        if r_dot_r < residual_tol:
            break
    return x

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.t = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = torch.tanh(self.fc_1(input))
        policy = F.softmax(self.fc_2(x))

        return policy

    @classmethod
    def train_model(cls, net, transitions, k):
        states, actions, rewards, masks = transitions
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)

        policy = net(states)
        policy = policy.view(-1, net.num_outputs)
        policy_action = (policy * actions.detach()).sum(dim=1)

        old_policy = net(states).detach()
        old_policy = old_policy.view(-1, net.num_outputs)
        old_policy_action = (old_policy * actions.detach()).sum(dim=1)

        surrogate_loss = ((policy_action / old_policy_action) * rewards).mean()

        surrogate_loss_grad = torch.autograd.grad(surrogate_loss, net.parameters())
        surrogate_loss_grad = flat_grad(surrogate_loss_grad)

        step_dir = conjugate_gradient(net, states, surrogate_loss_grad.data)

        params = flat_params(net)
        shs = (step_dir * fisher_vector_product(net, states, step_dir)).sum(0, keepdim=True)
        step_size = torch.sqrt((2 * max_kl) / shs)[0]
        full_step = step_size * step_dir

        fraction = 1.0
        for _ in range(10):
            new_params = params + fraction * full_step
            update_model(net, new_params)
            policy = net(states)
            policy = policy.view(-1, net.num_outputs)
            policy_action = (policy * actions.detach()).sum(dim=1)
            surrogate_loss = ((policy_action / old_policy_action) * rewards).mean()

            kl = kl_divergence(policy, old_policy)
            kl = kl.mean()

            if kl < max_kl:
                break
            fraction = fraction * 0.5

        return -surrogate_loss

    def get_action(self, input):
        policy = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action
