from qlearning.train import train, train_with_e_learning
from qlearning.models import Qnet, Enet
import gym
import numpy as np

batch_size = 32

if __name__ == "__main__":
    np.random.seed(42)
    seed_range = [np.random.randint(1000) for _ in range(5)]

    eps_params = {'exploration_fraction': 0.5,
                  'exploration_final_eps': 0.01}

    common_params = dict(gamma=0.99, write_logs=False,
                         target_type='standard_q_learning')
    act_type = 'epsilon_greedy'
    ucb = True

    set_weights = False
    zeros=True

    params = dict(eps_params=eps_params, e_lr=1e-4,
                  lr=1e-4, act_type=act_type)

    max_num_episodes = 200
    results = np.zeros((len(seed_range), max_num_episodes))

    for i, seed in enumerate(seed_range):
        env = gym.make('Acrobot-v1')
        model = Qnet(env.action_space.n,
                       env.observation_space.shape[0],
                       hidden_size=1024, num_hidden=2,
                       set_weights=set_weights, zeros=zeros, seed=seed
                       )
        e_model = Enet(env.action_space.n,
                       env.observation_space.shape[0],
                       hidden_size=1024, num_hidden=2, seed=seed)

        rews, num_episodes = train_with_e_learning(env,model, e_model,
                                   add_ucb=ucb,
                                   seed=seed,
                                   beta=100,
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

    filename = 'acrobot-test'
    if 'lll' in act_type:
        filename += '_lll'
    if ucb:
        filename += '_ucb'
    if set_weights:
        if zeros:
            filename += '_zeros'
        else:
            filename += '_ones'

    np.save('../results/dqn_environments/'+filename, results)