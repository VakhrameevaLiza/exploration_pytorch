import torch
import torch.nn as nn
import torch.nn.functional as F


class Qnet(nn.Module):
    def __init__(self, num_actions, input_dim,
                 num_hidden=2, hidden_size=512,
                 set_weights=False, zeros=True, seed=None,
                 activation_type='relu'):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        if activation_type == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(activation)
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        layers.append(nn.Linear(hidden_size, num_actions))
        self.net = nn.Sequential(*layers)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        if set_weights:
            self.set_weights(zeros=zeros)

    def set_weights(self, zeros=True):
        state_dict = self.net.state_dict()
        keys = sorted(state_dict.keys())
        if zeros:
            state_dict[keys[-2]] = torch.zeros_like(state_dict[keys[-2]])
            state_dict[keys[-1]] = torch.zeros_like(state_dict[keys[-1]])
            self.net.load_state_dict(state_dict)
        else:
            state_dict[keys[-2]] = torch.zeros_like(state_dict[keys[-2]])
            state_dict[keys[-1]] = torch.ones_like(state_dict[keys[-1]])
            self.net.load_state_dict(state_dict)

    def forward(self, x):
        out = self.net(x)
        return out


class Enet(nn.Module):
    def __init__(self, num_actions, input_dim,
                 num_hidden=2, hidden_size=512, seed=None,
                 activation_type ='relu'):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        if activation_type == 'tanh':
            activation = nn.Tanh()
        elif activation_type == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            activation = nn.ReLU()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(activation)
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        layers.append(nn.Linear(hidden_size, num_actions))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
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
