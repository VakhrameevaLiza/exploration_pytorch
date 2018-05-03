from qlearning.train import train
from qlearning.models import Qnet
import gym
import numpy as np

batch_size = 32

if __name__ == "__main__":
    np.random.seed(42)
    seed_range = [np.random.randint(1000) for _ in range(5)]

    eps_params = {'exploration_fraction': 0.5,
                  'exploration_final_eps': 0.01}

    common_params = dict(gamma=0.99, write_logs=True,
                         log_dir='logs/mountain_car',
                         target_type='standard_q_learning')
    params = dict(eps_params=eps_params, lr=1e-4, act_type='epsilon_greedy')

    max_num_episodes = 200
    results = np.zeros((len(seed_range), max_num_episodes))

    for i, seed in enumerate(seed_range):
        env = gym.make('CartPole-v0')
        model = Qnet(env.action_space.n,
                       env.observation_space.shape[0],
                       hidden_size=64, num_hidden=1, seed=seed,
                       )

        rews, num_episodes = train(env,model,
                                   seed=seed,
                                   replay_buffer_size=1e+5,
                                   batch_size=64,
                                   learning_starts_in_steps=500,
                                   max_steps=200*max_num_episodes,
                                   max_num_episodes=max_num_episodes,
                                   train_freq_in_steps=5,
                                   update_freq_in_steps=200,
                                   **common_params,
                                   **params)
        results[i] = rews

    #np.save('../results/dqn_environments/cartpole', results)