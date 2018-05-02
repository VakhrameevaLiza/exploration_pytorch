from qlearning.train import train, train_with_e_learning
from qlearning.models import Qnet, Enet
from tabular_environments.chain_environment import SimpleChain
import numpy as np

batch_size = 32

if __name__ == "__main__":
    np.random.seed(42)
    seed_range = [np.random.randint(1000) for _ in range(5)]
    dim_range = [5, 10, 15, 20, 25, 30]
    ucb_act_flags = [False, True]

    eps_params = {'exploration_fraction': 0.5,
                  'exploration_final_eps': 0.01}

    params = dict(gamma=0.99, gamma_E=0.9, write_logs=False,
                  target_type='standard_q_learning', eps_params=eps_params,
                  lr=1e-4, e_lr=1e-4, act_type='lll_epsilon_greedy')

    max_num_episodes = 1000

    for set_weight, zeros in [(False, False), (True, True), (True, False)]:
        for ucb in ucb_act_flags:
            results = np.zeros((len(seed_range), len(dim_range)))
            for i, seed in enumerate(seed_range):
                for j, dim in enumerate(dim_range):
                    print(ucb,i,j)
                    env = SimpleChain(dim)
                    model = Qnet(env.action_space.n,
                                   env.observation_space.shape[0],
                                   hidden_size=128, num_hidden=2,
                                   set_weights=set_weight, zeros=zeros)

                    e_model = Enet(env.action_space.n,
                                   env.observation_space.shape[0],
                                   hidden_size=128, num_hidden=2)

                    rews, num_episodes = train_with_e_learning(env,model, e_model,
                                               seed=seed,
                                               add_ucb=ucb,
                                               replay_buffer_size=1e+6,
                                               batch_size=64,
                                               learning_starts_in_steps=250,
                                               max_steps=(dim+9)*max_num_episodes,
                                               max_num_episodes=max_num_episodes,
                                               train_freq_in_steps=5,
                                               update_freq_in_steps=100,
                                               print_freq=None,
                                               **params)
                    results[i][j] = num_episodes
            dir = '../results/dqn_environments/deep_chain/'
            file_name = dir + 'lll_eps_greedy'
            if ucb:
                file_name += '_ucb'
            if set_weight:
                if zeros:
                    file_name += '_zeros'
                else:
                    file_name += '_ones'
            np.save(file_name, results)
