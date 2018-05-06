from qlearning.train import train
from qlearning.models import Qnet
import gym
import numpy as np
import os

batch_size = 32

if __name__ == "__main__":
    np.random.seed(42)
    seed_range = [np.random.randint(1000) for _ in range(3)]

    eps_params = {'exploration_fraction': 0.25,
                  'exploration_final_eps': 0.05}

    common_params = dict(gamma=0.99, write_logs=False,
                         log_dir='logs/mountain_car',
                         target_type='double_q_learning')

    params = dict(eps_params=eps_params, lr=1e-4, act_type='epsilon_greedy')

    max_num_episodes = 5000
    results = np.zeros((len(seed_range), max_num_episodes))
    all_history = []

    for i, seed in enumerate(seed_range):
        env = gym.make('MountainCar-v0')
        model = Qnet(env.action_space.n,
                       env.observation_space.shape[0],
                       hidden_size=512, num_hidden=2, seed=seed,
                       activation_type='relu')

        rews, num_episodes, history = train(env,model,
                                               seed=seed,
                                               replay_buffer_size=1e+5,
                                               batch_size=64,
                                               learning_starts_in_steps=500,
                                               max_steps=200*max_num_episodes,
                                               max_num_episodes=max_num_episodes,
                                               train_freq_in_steps=5,
                                               update_freq_in_steps=200,
                                               **common_params,
                                               **params,
                                               return_states=True)

        all_history.append(history)
        results[i] = rews

        dir = os.path.dirname(os.path.abspath(__file__))
        filename = 'mountain_car'
        np.save(dir+'/results/dqn_environments/'+filename, results)

        filename = 'mountain_car_history'
        np.save(dir+'/results/dqn_environments/'+filename, np.concatenate(all_history))

    #16:51
