import numpy as np
from helpers.convert_to_var_foo import convert_to_var


def epsilon_greedy_act(num_actions, state, model, eps_t, ucb=None, log_file=None):
    state_var = convert_to_var(state, add_dim=True)
    q_values = model.forward(state_var).data.numpy()[0]
    if ucb is not None:
        q_values += ucb
    if np.random.rand() < eps_t:
        action = np.random.randint(num_actions)
        flag='random'
    else:
        action = q_values.argmax()
        flag='argmax'
    if log_file is not None:
        s = str(action) + ',' + ','.join([str(q) for q in q_values]) + '\n'
        with open(log_file, 'a') as f:
            f.write(s)
    return action, flag
