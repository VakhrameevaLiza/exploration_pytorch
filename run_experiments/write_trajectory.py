from qlearning.train import train_tabular
from qlearning.models import Qnet, Enet
from tabular_environments.chain_environment import SimpleChain
import numpy as np
import os

batch_size = 32

if __name__ == "__main__":
    eps_params = {'exploration_fraction': 0.25,
                  'exploration_final_eps': 0.01}

    params = dict(gamma=0.99,
                  write_logs=False, target_type='standard_q_learning',
                  eps_params=eps_params, lr=1e-4,
                  act_type='epsilon_greedy')

    max_num_episodes = 400
    dim = 10
    env = SimpleChain(dim)

    model = Qnet(env.action_space.n,
                       env.observation_space.shape[0],
                       hidden_size=256, num_hidden=1,
                       set_weights=False)

    state_history, counters, state_ids = train_tabular(env, model,
                               replay_buffer_size=5000,
                               batch_size=256,
                               learning_starts_in_steps=400 ,
                               max_steps=(dim+9)*max_num_episodes,
                               train_freq_in_steps=5,
                               update_freq_in_steps=50,
                               print_freq=1,
                               **params,
                               )
    np.save('results/trajectories/state_history', state_history)
    np.save('results/trajectories/counters', counters)
    np.save('results/trajectories/state_ids', state_ids)
