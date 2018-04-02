import numpy as np
import matplotlib.pyplot as plt

from helpers.replay_buffer import ReplayBuffer
from helpers.chain_environment import SimpleChain
from helpers.shedules import LinearSchedule
from helpers.create_empty_directory import create_empty_directory
from helpers.plots import plot_q_func_and_visitations


def get_reward_addition(s_id, next_s_id, a, C, reward_shaping_type):
    if reward_shaping_type == 'count_based_state_action':
        add = 1 / np.sqrt(1 + C[s_id][a])
    elif reward_shaping_type == 'count_based_next_state_action':
        add = 1 / np.sqrt(1 + C[next_s_id][a])
    elif reward_shaping_type == 'count_based_state':
        add = 1 / np.sqrt(1 + C[s_id].sum())
    elif reward_shaping_type == 'count_based_next_state':
        add = 1 / np.sqrt(1 + C[next_s_id].sum())
    else:
        add = 0
    return add


def get_action(Q, C, s_id, num_actions, eps_t, act_type, alpha):
    if np.random.rand() < eps_t:
        a = np.random.randint(num_actions)
    elif act_type == 'argmax':
        a = Q[s_id].argmax()
    elif act_type == 'ucb-1':
        n_s = max(1, C[s_id].sum())
        n_sa = np.maximum(C[s_id], 1e-6)
        ucb_1 = np.sqrt(2 * np.log(n_s) / n_sa)
        a = (Q[s_id] + alpha * ucb_1).argmax()
    elif act_type == 'ucb-2':
        ucb_2 = 1 / np.sqrt(C[s_id] + 1)
        a = (Q[s_id] + alpha * ucb_2).argmax()
    return a


def tabular_q_learning(env, num_episodes, gamma=0.99,
                       eps_params=None, lr=0.5,
                       act_type='argmax', reward_shaping_type=None,
                       alpha=1.0,
                       img_folder='images', print_logs=True,
                       num_good_episodes_to_break=100, seed=42
                       ):
    if img_folder:
        create_empty_directory(img_folder)

    dim_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    if seed:
        np.random.seed(seed)

    # define shedule of epsilon in epsilon-greedy exploration
    if eps_params is not None:
        schedule_timesteps = int(eps_params['exploration_fraction'] * num_episodes * (dim_states + 9))
        eps_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                     initial_p=1.0,
                                     final_p=eps_params['exploration_final_eps'])
    else:
        eps_shedule = None

    Q = np.random.rand(dim_states * num_actions).reshape((dim_states, num_actions)) / 10
    C = np.zeros((dim_states, num_actions))
    t = 0

    episode_rews = []

    for episode in range(num_episodes):
        s = env.reset()
        s_id = env.cur_state_id
        episode_C = np.zeros((dim_states, num_actions))
        episode_rews.append(0)

        while True:
            eps_t = eps_shedule.value(t) if eps_shedule else 0

            a = get_action(Q, C, s_id, num_actions, eps_t, act_type, alpha)
            next_s, r, done, _ = env.step(a)
            next_s_id = env.cur_state_id
            r_add = get_reward_addition(s_id, next_s_id, a, C, reward_shaping_type)

            target_Q = (r + r_add) + gamma * Q[next_s_id].max() * (1 - done)
            Q[s_id][a] += lr * (target_Q - Q[s_id][a])

            C[s_id][a] += 1
            episode_C[s_id][a] += 1
            episode_rews[-1] += r

            s = next_s
            s_id = next_s_id

            t += 1

            if done:
                if print_logs:
                    print('episode: {}, total reward: {}'.format(episode, episode_rews[-1]))
                if img_folder:
                    plot_q_func_and_visitations(episode_C, C, Q, episode, t, img_folder)
                break

        if len(episode_rews) > num_good_episodes_to_break:
            if np.mean(episode_rews[-1 * num_good_episodes_to_break:])==10:
                break

    return Q, C, episode_rews


if __name__ == '__main__':
    print('main!')
    eps_params = {'exploration_fraction': 0.25,
                  'exploration_final_eps': 0.001}
    print('-')
    env = SimpleChain(10)
    tabular_q_learning(env, 2000, gamma=0.95,
                        eps_params=None, lr=0.5,
                        act_type='ucb-1', reward_shaping_type=None,
                        alpha=1.0,
                        img_folder=None, print_logs=True,
                        num_good_episodes_to_break=100,
                        seed=51,
                        )