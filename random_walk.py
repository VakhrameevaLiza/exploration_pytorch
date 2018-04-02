import numpy as np
import matplotlib.pyplot as plt
from helpers.chain_environment import SimpleChain
from helpers.create_empty_directory import create_empty_directory
from helpers.plots import plot_q_func_and_visitations


def go_random_walk(env, img_folder, tol=1e-6, max_num_episodes=5000):
    create_empty_directory(img_folder)

    dim_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    C = np.zeros((dim_states, num_actions))
    Q = np.zeros((dim_states, num_actions))

    t = 0
    for episode in range(max_num_episodes):
        s = env.reset()
        s_id = env.cur_state_id
        episode_C = np.zeros((dim_states, num_actions))

        while True:
            a = np.random.randint(num_actions)
            next_s, r, done, _ = env.step(a)
            next_s_id = env.cur_state_id

            C[s_id][a] += 1
            episode_C[s_id][a] += 1
            t += 1

            s = next_s
            s_id = next_s_id
            if done:
                print(episode)
                if episode % 500 == 0:
                    plot_q_func_and_visitations(episode_C, C, Q, episode, t, img_folder)
                break

if __name__ == '__main__':
    env = SimpleChain(20)
    go_random_walk(env, 'logs/random_walk/chain/images', max_num_episodes=50000,)