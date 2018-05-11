import numpy as np
import torch
import torch.nn as nn
from helpers.convert_to_var_foo import convert_to_var


def dqn_loss(optimizer, model, target_model, batch, gamma,
                      target_type='standard_q_learning'
                     ):
    states_batch, actions_batch, rewards_batch, dones_batch, next_states_batch, _ = batch
    states_batch_var = convert_to_var(states_batch)
    actions_batch_var = convert_to_var(actions_batch[:, np.newaxis], astype='int64')
    rewards_batch_var = convert_to_var(rewards_batch)
    next_states_batch_var = convert_to_var(next_states_batch)
    dones_batch_var = convert_to_var(dones_batch)

    q_values = model.forward(states_batch_var).gather(1, actions_batch_var)

    if target_type == 'standard_q_learning':
        all_next_q_values  = target_model.forward(next_states_batch_var).detach()
        best_next_q_values = all_next_q_values.max(dim=1)[0]
    elif target_type == 'double_q_learning':
        all_next_q_values = target_model.forward(next_states_batch_var).detach()
        argmax = torch.max(model.forward(next_states_batch_var), dim=1)[1]
        best_next_q_values = all_next_q_values.gather(1, argmax.view((-1, 1)))[:,0]
    else:
        print('Unknown Q-learning target type')
        return

    best_next_q_values[dones_batch_var.byte()] = 0
    q_values_targets = rewards_batch_var + gamma * best_next_q_values

    mse_loss_func = nn.MSELoss()
    loss = mse_loss_func(q_values, q_values_targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data[0]


def sarsa_loss(optimizer, model, target_model, batch, gamma):
    states, actions, rewards, dones, next_states, next_actions = batch

    states = convert_to_var(states)
    actions = convert_to_var(actions[:, np.newaxis], astype='int64')
    rewards = convert_to_var(rewards)
    next_states = convert_to_var(next_states)
    next_actions = convert_to_var(next_actions[:, np.newaxis], astype='int64')
    dones = convert_to_var(dones)

    rewards = torch.zeros_like(rewards)

    e_values = model.forward(states).gather(1, actions)[:,0]
    next_e_values = target_model.forward(next_states).gather(1, next_actions).detach()[:,0]
    next_e_values[dones.byte()] = 0

    target_e_values = rewards + gamma * next_e_values
    mse_loss_func = nn.MSELoss()
    loss = mse_loss_func(e_values, target_e_values)
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    return loss.data[0]
