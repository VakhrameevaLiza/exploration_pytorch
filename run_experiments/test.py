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
                  lr=1e-4, e_lr=1e-4, act_type='epsilon_greedy')

    max_num_episodes = 1000
    seed = 42
    dim = 5
    env = SimpleChain(dim)
    model = Qnet(env.action_space.n,
                   env.observation_space.shape[0],
                   hidden_size=128, num_hidden=2,
                   set_weights=False)

    e_model = Enet(env.action_space.n,
                   env.observation_space.shape[0],
                   hidden_size=128, num_hidden=2)

    rews, num_episodes = train_with_e_learning(env,model, e_model,
                               seed=seed,
                               add_ucb=True,
                               replay_buffer_size=1e+6,
                               batch_size=64,
                               learning_starts_in_steps=250,
                               max_steps=(dim+9)*max_num_episodes,
                               max_num_episodes=max_num_episodes,
                               train_freq_in_steps=5,
                               update_freq_in_steps=100,
                               print_freq=1,
                               do_pretraining=True,
                               **params)
