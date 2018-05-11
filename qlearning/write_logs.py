from helpers.convert_to_var_foo import convert_to_var
from helpers.plots import plot_q_func_and_visitations
import numpy as np
import torch


def write_tensorboard_tabular_logs(lcl):
    lcl['writer'].add_scalar('dqn/metrics/loss', lcl['loss'], lcl['t'])
    all_states_var = convert_to_var(lcl['env'].get_all_states())

    plot_q_func = lcl['act_type'] in ['epsilon_greedy', 'ucb']

    if plot_q_func:
        all_q_values = lcl['model'].forward(all_states_var)

        if torch.cuda.is_available():
            all_q_values = all_q_values.cpu().data.numpy()
        else:
            all_q_values = all_q_values.data.numpy()

        all_e_values = lcl['e_model'].forward(all_states_var)

        if torch.cuda.is_available():
            all_e_values = all_e_values.cpu().data.numpy()
        else:
            all_e_values = all_e_values.data.numpy()

        for i in range(lcl['n_all_states']):
            if (2 <= i < lcl['n_all_states'] - 2) and lcl['n_all_states'] >= 10:
                continue
            else:
                lcl['writer'].add_scalars('dqn/q_values/state_{}'.format(i + 1),
                                          {'action_right': all_q_values[i][1],
                                           'action_left': all_q_values[i][0]},
                                          lcl['t'])

    lcl['writer'].add_scalar('dqn/metrics/count_good_reward', lcl['count_good_rewards'], lcl['t'])
    lcl['writer'].add_scalar('dqn/metrics/eps_t', lcl['eps_t'], lcl['t'])
    lcl['writer'].add_scalar('dqn/metrics/rew_addition', lcl['rew_addition'], lcl['t'])
    if lcl['done']:
        lcl['writer'].add_scalar('dqn/metrics/sum_rewards_per_episode',
                                 lcl['sum_rewards_per_episode'][-1],
                                 lcl['num_episodes'])

    if lcl['t'] > lcl['learning_starts_in_steps'] and lcl['t'] % lcl['train_freq_in_steps'] == 0:
        lcl['writer'].add_scalar('dqn/metrics/rewards_batch', np.sum(lcl['batch'][2]), lcl['t'])

    if lcl['num_episodes'] % lcl['plot_freq'] == 0 and lcl['done']:
        if plot_q_func:
            plot_q_func_and_visitations(lcl['episode_visitations'],
                                        lcl['state_action_count'], all_q_values, all_e_values,
                                        lcl['num_episodes'], lcl['t'], lcl['images_directory'])


def write_tensorboard_logs(lcl):
    lcl['writer'].add_scalar('dqn/metrics/loss', lcl['loss'], lcl['t'])
    lcl['writer'].add_scalar('dqn/metrics/eps_t', lcl['eps_t'], lcl['t'])
    if lcl['done']:
        lcl['writer'].add_scalar('dqn/metrics/sum_rewards_per_episode',
                                 lcl['sum_rewards_per_episode'][-1],
                                 lcl['num_episodes'])