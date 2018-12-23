import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_adv = nn.Linear(128, num_outputs)
        self.fc_val = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc(x))
        adv = self.fc_adv(x)
        adv = adv.view(-1, self.num_outputs)
        val = self.fc_val(x)
        val = val.view(-1, 1)

        qvalue = val + (adv - adv.mean(dim=1, keepdim=True))
        return qvalue

class LocalModel(Model):
    def __init__(self, num_inputs, num_outputs):
        super(LocalModel, self).__init__(num_inputs, num_outputs)

    def pull_from_global_model(self, global_model):
        self.load_state_dict(global_model.state_dict())

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]