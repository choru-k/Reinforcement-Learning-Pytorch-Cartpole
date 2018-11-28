import torch
import torch.nn as nn
import torch.nn.functional as F

def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.1)

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_actor = nn.Linear(128, num_outputs)

        self.fc3 = nn.Linear(num_inputs, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc_critic = nn.Linear(128, 1)

        set_init([self.fc1, self.fc2, self.fc_actor, self.fc3, self.fc4, self.fc_critic])

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.fc_actor(x))

        y = F.relu(self.fc3(input))
        y = F.relu(self.fc4(y))
        value = self.fc_critic(y)
        return policy, value


class GlobalModel(Model):
    def __init__(self, num_inputs, num_outputs):
        super(GlobalModel, self).__init__(num_inputs, num_outputs)


class LocalModel(Model):
    def __init__(self, num_inputs, num_outputs):
        super(LocalModel, self).__init__(num_inputs, num_outputs)

    def push_to_global_model(self, batch, global_model, global_optimizer, args):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).long()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        policy, value = self.forward(states[0])
        _, last_value = self.forward(next_states[-1])

        running_returns = last_value[0]
        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + args.gamma * running_returns * masks[t]

        pred = running_returns
        td_error = pred - value[0]

        log_policy = torch.log(policy[0] + 1e-5)[actions[0]]
        loss1 = - log_policy * td_error.item()
        loss2 = F.mse_loss(value[0], pred.detach())
        entropy = torch.log(policy + 1e-5) * policy
        loss = loss1 + loss2 - 0.01 * entropy.sum()

        global_optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(self.parameters(), global_model.parameters()):
            gp._grad = lp.grad
        global_optimizer.step()

    def pull_from_global_model(self, global_model):
        self.load_state_dict(global_model.state_dict())
