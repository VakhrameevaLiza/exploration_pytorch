import numpy as np
import os
import torch
import gym

from trpo.agent import TRPOAgent
from trpo.run import run_trpo

env = gym.make('MountainCar-v0')

np.random.seed(42)
seed_range = [np.random.randint(1000) for _ in range(3)]
max_steps = 500
num_episodes_per_rollout = 50

results = np.zeros((len(seed_range), max_steps))

for i, seed in enumerate(seed_range):
    agent = TRPOAgent(state_shape=env.observation_space.shape,
                      n_actions=env.action_space.n,
                      hidden_size=128)

    rewards = run_trpo(env, agent,
                       seed=seed,
                       max_steps=max_steps,
                       num_episodes_per_rollout=num_episodes_per_rollout,
                       print_flag=True)

    results[i] = rewards

    filename = 'trpo_moutain_car'
    dir = os.path.dirname(os.path.abspath(__file__))
    np.save(dir + '/results/trpo_environments/' + filename, results)