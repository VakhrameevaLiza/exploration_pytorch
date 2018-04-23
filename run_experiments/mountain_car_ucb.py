from qlearning.train import train, train_with_e_learning
from qlearning.models import Qnet, Enet
import gym
import numpy as np

batch_size = 32

if __name__ == "__main__":
    np.random.seed(42)
    seed_range = [np.random.randint(1000) for _ in range(5)]

    eps_params = {'exploration_fraction': 0.25,
                  'exploration_final_eps': 0.01}

    common_params = dict(gamma=0.99, write_logs=True,
                         log_dir='logs/mountain_car',
                         target_type='standard_q_learning')
    params = dict(eps_params=eps_params, lr=1e-4, act_type='epsilon_greedy')

    max_num_episodes = 7500
    results = np.zeros((len(seed_range), max_num_episodes))

    for i, seed in enumerate(seed_range):
        env_name = 'MountainCar-v0'
        env = gym.make(env_name)
        model = Qnet(env.action_space.n,
                     env.observation_space.shape[0],
                     hidden_size=256, num_hidden=2)

        e_model = Enet(env.action_space.n,
                       env.observation_space.shape[0],
                       hidden_size=256, num_hidden=2)

        rews, num_episodes = train_with_e_learning(
                                   env, model, e_model,
                                   add_ucb=True,
                                   seed=seed,
                                   replay_buffer_size=1e+6,
                                   batch_size=256,
                                   learning_starts_in_steps=500,
                                   max_steps=200*max_num_episodes,
                                   max_num_episodes=max_num_episodes,
                                   train_freq_in_steps=5,
                                   update_freq_in_steps=50,
                                   **common_params,
                                   **params)
        results[i] = rews

    np.save(env_name+'_ucb', results)