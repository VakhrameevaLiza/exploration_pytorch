import numpy as np
import matplotlib.pyplot as plt

from helpers.replay_buffer import ReplayBuffer
from helpers.chain_environment import SimpleChain
from helpers.shedules import LinearSchedule
from helpers.create_empty_directory import create_empty_directory
from helpers.plots import plot_q_func_and_visitations


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

    Q = np.random.rand(dim_states*num_actions).reshape((dim_states, num_actions))#np.ones((dim_states, num_actions))
    C = np.zeros((dim_states, num_actions))
    t = 0

    episode_rews = []

    for episode in range(num_episodes):
        episode_C = np.zeros((dim_states, num_actions))
        episode_rews.append(0)

        s = env.reset()
        s_id = env.cur_state_id
        eps_t = eps_shedule.value(t) if eps_shedule else 0
        if np.random.rand() < eps_t:
            a = np.random.randint(num_actions)
        else:
            a = Q[s_id].argmax()

        while True:
            t += 1
            eps_t = eps_shedule.value(t) if eps_shedule else 0
            next_s, r, done, _ = env.step(a)
            r = 1 / np.sqrt(1 + C[s_id][a])
            next_s_id = env.cur_state_id

            if np.random.rand() < eps_t:
                next_a = np.random.randint(num_actions)
            else:
                next_a = Q[next_s_id].argmax()

            target_E = r + gamma * Q[next_s_id][next_a] * (1 - done)
            Q[s_id][a] += lr * (target_E - Q[s_id][a])

            C[s_id][a] += 1
            episode_C[s_id][a] += 1
            episode_rews[-1] += r

            s = next_s
            s_id = next_s_id
            a = next_a

            if done:
                if print_logs:
                    print('episode: {}, total reward: {}'.format(episode, episode_rews[-1]))
                if img_folder and episode % 500 == 0:
                    plot_q_func_and_visitations(episode_C, C, Q, episode, t, img_folder)
                break

        if len(episode_rews) > num_good_episodes_to_break:
            if np.mean(episode_rews[-1 * num_good_episodes_to_break:])==10:
                break

    return Q, C, episode_rews


if __name__ == '__main__':
    eps_params = {'exploration_fraction': 0.05,
                  'exploration_final_eps': 0.001}
    print('-')
    env = SimpleChain(20)
    tabular_q_learning(env, 50000, gamma=0.95,
                        eps_params=None, lr=0.1,
                        img_folder='logs/tabular_sarsa_exploration/chain', print_logs=True,
                        seed=51,
                        )