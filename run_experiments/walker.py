import numpy as np
import os
import torch
import gym

from trpo.agent import TRPOAgent
from trpo.run import run_trpo

env = gym.make('Walker2d-v2')

agent = TRPOAgent(state_shape=env.observation_space.shape,
                  action_shape=env.action_space.shape,
                  hidden_size=50,
                  )
seed = 42
results = run_trpo(env, agent,
                   seed=seed,
                   max_steps=100,
                   print_flag=True,
                   log_dir='../logs/walker')

result_dir = '../results/trpo_environments/'

filename = 'walker'
dir = os.path.dirname(os.path.abspath(__file__))
np.save(dir + '/results/dqn_environments/' + filename, results)

torch.save(agent.policy.state_dict(), result_dir+'walker_model')
