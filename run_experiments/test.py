from qlearning.train import train_tabular_with_e_learning
from qlearning.models import Qnet, Enet
from tabular_environments.chain_environment import SimpleChain
import numpy as np
import os

batch_size = 32

if __name__ == "__main__":
    eps_params = {'exploration_fraction': 0.25,
                  'exploration_final_eps': 0.01}

    params = dict(gamma=0.99, gamma_E=0.9,
                  write_logs=True, target_type='standard_q_learning',
                  eps_params=eps_params, lr=1e-4, e_lr=1e-5,
                  act_type='epsilon_greedy')

    max_num_episodes = 1000
    dim = 10
    env = SimpleChain(dim)

    model = Qnet(env.action_space.n,
                       env.observation_space.shape[0],
                       hidden_size=256, num_hidden=1,
                       set_weights=False)

    e_model = Enet(env.action_space.n,
                   env.observation_space.shape[0],
                   activation_type='relu',
                   hidden_size=512, num_hidden=2)

    rews, num_episodes = train_tabular_with_e_learning(env, model, e_model,
                                                       replay_buffer_size=5000,
                                                       batch_size=256,
                                                       learning_starts_in_steps=100 ,
                                                       max_steps=(dim+9)*max_num_episodes,
                                                       max_num_episodes=1000,
                                                       train_freq_in_steps=5,
                                                       update_freq_in_steps=50,
                                                       print_freq=1,
                                                       do_pretraining=True,
                                                       **params)
