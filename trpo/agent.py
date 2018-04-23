from trpo.models import DiscretePolicy, ContinuousPolicy, ValueFunction
from torch.autograd import Variable
import numpy as np
import torch
from scipy.stats import norm


class TRPOAgent:
    def __init__(self, state_shape, n_actions=None, action_shape=None, hidden_size=250):
        '''
        Here you should define your model
        You should have LOG-PROBABILITIES as output because you will need it to compute loss
        We recommend that you start simple:
        use 1-2 hidden layers with 100-500 units and relu for the first try
        '''
        self.discrete_type = n_actions is not None
        if self.discrete_type:
            self.policy = DiscretePolicy(n_actions, state_shape[0],
                                         hidden_size=hidden_size)
        else:
            self.policy = ContinuousPolicy(action_shape[0], state_shape[0],
                                           hidden_size=hidden_size)

        self.values = ValueFunction(state_shape, hidden_size=250)

    def get_values(self, states):
        return self.values.forward(states)

    def get_log_probs(self, states):
        return self.policy.forward(states)

    def get_probs(self, states):
        return torch.exp(self.policy.forward(states))

    def act(self, obs, sample=True):
        if self.discrete_type:
            probs = self.get_probs(Variable(torch.FloatTensor([obs]))).data.numpy()
            n_actions = probs.shape[1]
            if sample:
                action = int(np.random.choice(n_actions, p=probs[0]))
            else:
                action = int(np.argmax(probs[0]))
            return action, probs[0][action], probs[0]
        else:
            mu, logvar = self.policy.forward(Variable(torch.FloatTensor([obs])))
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

