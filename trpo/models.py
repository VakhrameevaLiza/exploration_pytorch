import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from helpers.replay_buffer import ReplayBuffer


class DiscretePolicy(nn.Module):
    def __init__(self, num_actions, input_dim, num_hidden=2, hidden_size=512):
        super().__init__()
        layers = list()
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_actions))
        layers.append(nn.LogSoftmax(dim=-1))
        self.model = nn.Sequential(*layers)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def forward(self, states):
        log_probs = self.model(states)
        return log_probs

    def get_probs_for_actions(self, states, actions):
        log_probs = self.model(states)
        probs_all = torch.exp(log_probs)
        batch_size = states.shape[0]
        if torch.cuda.is_available():
            probs_for_actions = probs_all[torch.arange(0, batch_size, out=torch.cuda.LongTensor()), actions]
        else:
            probs_for_actions = probs_all[torch.arange(0, batch_size, out=torch.LongTensor()), actions]
        return probs_for_actions


class ContinuousPolicy(nn.Module):
    def __init__(self, action_dim, input_dim, num_hidden=1, hidden_size=512):
        super().__init__()
        layers = list()
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        for i in range(num_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 2 * action_dim))
        self.model = nn.Sequential(*layers)
        self.action_dim = action_dim
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def forward(self, states):
        out = self.model(states)
        mu = out[:, :self.action_dim]
        logvar = out[:, self.action_dim:]
        return mu, logvar

    def get_probs_for_actions(self, states, actions):
        out = self.model(states)

        mu = out[:, :self.action_dim]
        logvar = out[:, self.action_dim:]
        var = torch.exp(logvar)

        batch_size, d = actions.shape

        coef = 1 / ((2 * np.pi)**d * torch.prod(var, dim=1)).pow(0.5)
        exp = torch.exp(-0.5 * torch.sum((actions - mu).pow(2)/var, dim=1))
        probs_for_actions = coef * exp
        return probs_for_actions


class ValueFunction(nn.Module):
    def __init__(self, state_shape, hidden_size=250):
        nn.Module.__init__(self)
        self.model = nn.Sequential(nn.Linear(state_shape[-1], hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, 1))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    def forward(self, states):
        values = self.model(states)
        return values


class Enet(nn.Module):
    def __init__(self, num_actions, input_dim,
                 num_hidden=2, hidden_size=512, seed=None,
                 activation_type ='relu'):
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


class ELearningParameters:
    def __init__(self, model, beta=1e+4, e_lr=1e-4, gamma_E=0.9,
                 train_freq=5, update_freq=200, batch_size=64,
                 replay_buffer_size=1e+5, seed=None):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.t = 0
        self.beta = beta
        self.e_lr = e_lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), e_lr)
        self.gamma_E = gamma_E
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.update_freq = update_freq
        self.learning_starts = 500

        self.replay_buffer = ReplayBuffer(replay_buffer_size, seed)
