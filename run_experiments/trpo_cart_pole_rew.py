import numpy as np
import os
import torch
import gym

from trpo.agent import TRPOAgent
from trpo.run import run_trpo, run_trpo_with_e_learning
from trpo.models import Enet, ELearningParameters

env = gym.make('CartPole-v0')

np.random.seed(42)
seed_range = [np.random.randint(1000) for _ in range(3)]
max_steps = 40
num_episodes_per_rollout = 10

results = np.zeros((len(seed_range), max_steps))

for i, seed in enumerate(seed_range):
    agent = TRPOAgent(state_shape=env.observation_space.shape,
                      n_actions=env.action_space.n,
                      hidden_size=128)

    e_model = Enet(env.action_space.n,
                   env.observation_space.shape[0],
                   hidden_size=512, num_hidden=2, seed=seed)

    e_learning = ELearningParameters(e_model, e_lr=1e-4, gamma_E=0.9, beta=1e3,
                                     train_freq=5, update_freq=200, batch_size=64)

    rewards = run_trpo_with_e_learning(env, agent, e_learning,
                                       seed=seed,
                                       max_steps=max_steps,
                                       num_episodes_per_rollout=num_episodes_per_rollout,
                                       print_flag=True)

    results[i] = rewards

    filename = 'trpo_cart_pole_rew'
    dir = os.path.dirname(os.path.abspath(__file__))
    np.save(dir + '/results/trpo_environments/' + filename, results)