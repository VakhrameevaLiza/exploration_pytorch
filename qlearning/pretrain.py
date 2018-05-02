import numpy as np
import torch
import torch.nn as nn
from helpers.convert_to_var_foo import convert_to_var


def pretrain(model, all_states, num_actions,
             max_steps, eps, writer):
    n_states = all_states.shape[0]
    target = convert_to_var(np.zeros((n_states,num_actions)))
    all_states = convert_to_var(all_states)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    mse = nn.MSELoss()
    for t in range(max_steps):
        q = model.forward(all_states)

        loss = mse(q, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(n_states):
            if (2 <= i < n_states - 2) and n_states >= 10:
                continue
            else:
                writer.add_scalars('dqn/q_values/state_{}'.format(i + 1), {'action_right': q[i][1],
                                                                           'action_left': q[i][0]}, t)
        if loss.data.numpy()[0] < eps:
            return t

    return max_steps
