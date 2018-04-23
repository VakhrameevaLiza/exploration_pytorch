import gym
import torch
import numpy as np
from helpers.convert_to_var_foo import convert_to_var

from trpo.agent import TRPOAgent
from trpo.run import run_trpo

env = gym.make('Swimmer-v2')

agent = TRPOAgent(state_shape=env.observation_space.shape,
                  action_shape=env.action_space.shape,
                  hidden_size=50,
                  )

result_dir = '../results/trpo_environments/'
agent.policy.load_state_dict(torch.load(result_dir+'swimmer_model'))

num_steps = 10000

obs = env.reset()
for i in range(num_steps):
    env.render()
    action = agent.act(obs)[0]
    print(action)
    obs, r, done, _ = env.step(action)
    if done:
        obs = env.reset()