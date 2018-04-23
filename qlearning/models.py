import torch
import torch.nn as nn
import torch.nn.functional as F


class Qnet(nn.Module):
    def __init__(self, num_actions, input_dim,
                 num_hidden=2, hidden_size=512):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class Enet(nn.Module):
    def __init__(self, num_actions, input_dim,
                 num_hidden=2, hidden_size=512):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_actions))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.set_weights()


    def set_weights(self):
        state_dict = self.net.state_dict()
        keys = sorted(state_dict.keys())
        state_dict[keys[-2]] = torch.zeros_like(state_dict[keys[-2]])
        state_dict[keys[-1]] = torch.zeros_like(state_dict[keys[-1]])
        self.net.load_state_dict(state_dict)


    def forward(self, x):
        out = self.net(x)
        return out
