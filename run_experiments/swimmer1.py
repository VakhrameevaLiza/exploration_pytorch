import numpy as np
import torch
import gym

from trpo.agent import TRPOAgent
from trpo.run import run_trpo

env = gym.make('Swimmer-v2')

agent = TRPOAgent(state_shape=env.observation_space.shape,
                  action_shape=env.action_space.shape,
                  hidden_size=50,
                  )
seed = 42
resutls = run_trpo(env, agent,
                   seed=seed,
                   max_steps=50,
                   print_flag=True,
                   log_dir='../logs/swimmer')

result_dir = '../results/trpo_environments/'
env_name = 'swimmer_rewards'
np.save(result_dir+env_name, resutls)
torch.save(agent.policy.state_dict(), result_dir+'swimmer_model')
