import torch
from torch.autograd import Variable
import numpy as np


def get_loss(agent, observations, actions, cummulative_returns,
             old_probs_for_actions):
    # Compute surrogate loss, aka importance-sampled policy gradient
    probs_for_actions = agent.policy.get_probs_for_actions(observations, actions)

    values = agent.get_values(observations)[:, 0]
    advantage = cummulative_returns - values

    #print('Loss')
    #print('probs_for_actions', probs_for_actions.shape)
    #print('old_probs_for_actions', old_probs_for_actions.shape)
    #print('advantage', advantage.shape)

    Loss = -1 * torch.mean(probs_for_actions / old_probs_for_actions * advantage)
    MSELoss = torch.nn.MSELoss()

    return Loss, MSELoss(values, cummulative_returns)


def get_discrete_kl(agent, observations, old_probs):
    #print('get_discrete_kl')
    old_log_probs = torch.log(old_probs + 1e-10)
    log_probs = agent.get_log_probs(observations)
    #print('old_log_probs', old_log_probs.shape)
    #print('log_probs', log_probs.shape)
    kl = torch.mean(torch.sum(old_probs * (old_log_probs - log_probs), dim=1))

    assert kl.shape == torch.Size([1])
    assert (kl > -0.0001).all() and (kl < 10000).all()
    return kl


def get_normal_kl(agent, observations, old_parameters):
    old_mu, old_logvar = old_parameters
    mu, logvar = agent.policy.forward(observations)
    var = torch.exp(logvar)
    old_var = torch.exp(old_logvar)
    kl = torch.mean(torch.sum(logvar - old_logvar + old_var / var
                              - 1 + (old_mu - mu).pow(2) / torch.exp(logvar), dim=1))
    kl *= 0.5
    assert kl.shape == torch.Size([1])
    assert (kl > -0.0001).all() and (kl < 10000).all()
    return kl


def get_discrete_entropy(agent, observations):
    observations = Variable(torch.FloatTensor(observations))

    batch_size = observations.shape[0]
    log_probs_all = agent.get_log_probs(observations)
    probs_all = torch.exp(log_probs_all)

    entropy = torch.sum(-probs_all * log_probs_all) / batch_size

    assert entropy.shape == torch.Size([1])
    return entropy


def get_normal_entropy(agent, observations):
    observations = Variable(torch.FloatTensor(observations))
    mu, logvar = agent.policy.forward(observations)
    var = torch.exp(logvar)
    entropy = 0.5 * torch.mean(
                          torch.sum(torch.log(2 * np.pi * np.e * var), dim=1))
    return entropy
