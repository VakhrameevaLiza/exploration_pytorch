import numpy as np
from helpers.convert_to_var_foo import convert_to_var
import torch

def epsilon_greedy_act(num_actions, state, model, eps_t, ucb=None, log_file=None):
    state_var = convert_to_var(state, add_dim=True)

    q_values = model.forward(state_var)#.cpu().data.numpy()[0]

    if ucb is not None:
        q_values += convert_to_var(ucb, add_dim=True)
    if np.random.rand() < eps_t:
        action = np.random.randint(num_actions)
    else:
        action = torch.argmax(q_values, dim=1)[0] #q_values.argmax()
    return action


def safe_log(values, base=np.e):
    eps = 1e-10 # avoid zero in log
    if sum(values < 1e-8) > 0:
        values += eps
    return np.log(values) / np.log(base)


def lll_epsilon_greedy_act(num_actions, state, model, e_model, e_lr,
                           eps_t, ucb=None, log_file=None):
    state_var = convert_to_var(state, add_dim=True)
    q_values = model.forward(state_var).data.numpy()[0]
    e_values = e_model.forward(state_var).data.numpy()[0]
    probs = np.ones_like(q_values) * eps_t / num_actions

    if ucb is not None:
        q_values += ucb

    probs[q_values.argmax()] += 1 - eps_t
    log_prob = safe_log(probs)
    log_log_E = safe_log(safe_log(e_values, 1 - e_lr) + np.log(2) / np.log(1-e_lr) )

    action = int(np.argmax(log_prob - log_log_E))

    return action
