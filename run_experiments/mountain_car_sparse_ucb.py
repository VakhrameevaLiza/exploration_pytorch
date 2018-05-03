from qlearning.train import train, train_with_e_learning
from qlearning.models import Qnet, Enet
import os
import numpy as np
from sparse_environments.sparse_environments import SparseAcrobot, SparseMountainCar
batch_size = 32

if __name__ == "__main__":
    np.random.seed(42)
    seed_range = [np.random.randint(1000) for _ in range(3)]

    eps_params = {'exploration_fraction': 0.25,
                  'exploration_final_eps': 0.01}

    common_params = dict(gamma=0.99, write_logs=False,
                         target_type='standard_q_learning')
    act_type = 'epsilon_greedy'
    ucb = True

    set_weights = False
    zeros=True

    params = dict(eps_params=eps_params, e_lr=1e-4,
                  lr=1e-4, act_type=act_type)

    max_num_episodes = 3000
    results = np.zeros((len(seed_range), max_num_episodes))

    for i, seed in enumerate(seed_range):
        env = SparseMountainCar()
        model = Qnet(env.action_space.n,
                       env.observation_space.shape[0],
                       hidden_size=512, num_hidden=1,
                       set_weights=set_weights, zeros=zeros, seed=seed
                       )
        e_model = Enet(env.action_space.n,
                       env.observation_space.shape[0],
                       hidden_size=512, num_hidden=1, seed=seed)

        rews, num_episodes = train_with_e_learning(env,model, e_model,
                                   add_ucb=ucb,
                                   seed=seed,
                                   beta=1000,
                                   replay_buffer_size=1e+5,
                                   batch_size=64,
                                   learning_starts_in_steps=500,
                                   max_steps=200*max_num_episodes,
                                   max_num_episodes=max_num_episodes,
                                   train_freq_in_steps=5,
                                   update_freq_in_steps=200,
                                   print_freq=10,
                                   **common_params,
                                   **params)
        results[i] = rews

        filename = 'mountain_car_sparse'
        dir = os.path.dirname(os.path.abspath(__file__))
        if ucb:
            filename += '_ucb'
        if set_weights:
            if zeros:
                filename += '_zeros'
            else:
                filename += '_ones'

        np.save(dir+'/results/dqn_environments/'+filename, results)