from trpo.models import DiscretePolicy, ContinuousPolicy, ValueFunction, Enet
from helpers.convert_to_var_foo import convert_to_var
import numpy as np
import torch
from scipy.stats import norm
import copy


class TRPOAgent:
    def __init__(self, state_shape, n_actions=None, action_shape=None,
                 hidden_size=250):
        self.discrete_type = n_actions is not None
        self.e_model = None
        if self.discrete_type:
            self.policy = DiscretePolicy(n_actions, state_shape[0],
                                         hidden_size=hidden_size)
        else:
            self.policy = ContinuousPolicy(action_shape[0], state_shape[0],
                                           hidden_size=hidden_size)
        self.values = ValueFunction(state_shape, hidden_size=hidden_size)

    def get_values(self, states):
        return self.values.forward(states)

    def get_log_probs(self, states):
        return self.policy.forward(states)

    def get_probs(self, states):
        return torch.exp(self.policy.forward(states))

    def act(self, obs, sample=True):
        if self.discrete_type:
            if torch.cuda.is_available():
                probs = self.get_probs(convert_to_var(obs, add_dim=True)).cpu().data.numpy()
            else:
                probs = self.get_probs(convert_to_var(obs, add_dim=True)).data.numpy()

            n_actions = probs.shape[1]
            if sample:
                action = int(np.random.choice(n_actions, p=probs[0]))
            else:
                action = int(np.argmax(probs[0]))
            return action, probs[0][action], probs[0]
        else:
            mu, logvar = self.policy.forward(convert_to_var(obs, add_dim=True))
            if torch.cuda.is_available():
                mu = mu.cpu().data.numpy()
                logvar = logvar.cpu().data.numpy()
            else:
                mu = mu.data.numpy()
                logvar = logvar.data.numpy()
            std = np.exp(0.5 * logvar)
            if sample:
                action_shape = mu.shape[1]
                eps = np.random.randn(action_shape)
            else:
                eps = np.zeros_like(mu)
            action = mu + eps * std
            action_prob = norm.pdf(eps).prod()
        return action[0], action_prob, mu[0], logvar[0]

