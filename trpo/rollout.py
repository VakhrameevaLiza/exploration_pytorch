from trpo.utils import get_cummulative_returns
from trpo.loss import sarsa_loss
from helpers.convert_to_var_foo import convert_to_var
import numpy as np
import copy
import torch

def rollout(env, agent, num_episodes):
    paths = []
    total_timesteps = 0
    for episode_i in range(num_episodes):
        obervations, actions, rewards, probs_for_actions = [], [], [], []
        mus, logvars, policies = [], [], []
        obervation = env.reset()
        while True:
            if agent.discrete_type:
                action, action_prob, probs = agent.act(obervation)
                policies.append(probs)
            else:
                action, action_prob, mu, logvar = agent.act(obervation)
                mus.append(mu)
                logvars.append(logvar)

            obervations.append(obervation)
            actions.append(action)
            probs_for_actions.append(action_prob)

            obervation, reward, done, _ = env.step(action)
            rewards.append(reward)
            total_timesteps += 1
            if done:
                path = {"observations": np.array(obervations),
                        "policies":  np.array(policies),
                        "mus": np.array(mus),
                        "logvars": np.array(logvars),
                        "probs_for_actions": np.array(probs_for_actions),
                        "actions": np.array(actions),
                        "rewards": np.array(rewards),
                        "cumulative_returns":get_cummulative_returns(rewards),
                       }
                paths.append(path)
                break
    return paths


def rollout_with_e_learning(env, agent, e_learning, num_episodes):

    e_optimizer = e_learning.optimizer
    learning_starts = e_learning.learning_starts
    train_freq = e_learning.train_freq
    update_freq = e_learning.update_freq
    gamma_E = e_learning.gamma_E
    batch_size = e_learning.batch_size
    beta = e_learning.beta
    e_lr = e_learning.e_lr

    paths = []
    total_timesteps = 0

    for episode_i in range(num_episodes):
        observations, actions, rewards, rewards_, dones, probs_for_actions = [], [], [], [], [], []
        mus, logvars, policies = [], [], []
        observation = env.reset()

        step_i = 0
        while True:
            if agent.discrete_type:
                action, action_prob, probs = agent.act(observation)
                policies.append(probs)
            else:
                action, action_prob, mu, logvar = agent.act(observation)
                mus.append(mu)
                logvars.append(logvar)

            observations.append(observation)
            actions.append(action)
            probs_for_actions.append(action_prob)

            observation, reward, done, _ = env.step(action)
            if torch.cuda.is_available():
                e_values = e_learning.model.forward(convert_to_var(observation)).cpu().data.numpy()
            else:
                e_values = e_learning.model.forward(convert_to_var(observation)).data.numpy()
            cnt = np.log(e_values) / np.log(1 - e_lr) + np.log(2) / np.log(1 - e_lr)
            reward_ = reward + beta * 1 / (cnt[action]+1e-6)
            #print(beta * 1 / (cnt[action]+1e-6))
            rewards.append(reward)
            rewards_.append(reward_)
            dones.append(done)

            if step_i > 0:
                e_learning.replay_buffer.add(observations[-2], actions[-2],
                                             rewards[-2], dones[-2],
                                             observations[-1], actions[-1])
            if e_learning.t > learning_starts:
                if step_i > 0:
                    batch = [np.array([observations[-2]]), np.array([actions[-2]]),
                             np.array([rewards[-2]]), np.array([dones[-2]]),
                             np.array([observations[-1]]), np.array([actions[-1]])]
                    sarsa_loss(e_optimizer, e_learning.model, e_learning.target_model,
                               batch, gamma_E)
                if e_learning.t % train_freq:
                    batch = e_learning.replay_buffer.sample(batch_size)
                    sarsa_loss(e_optimizer, e_learning.model, e_learning.target_model,
                               batch, gamma_E)
                if e_learning.t % update_freq:
                    e_learning.target_model = copy.deepcopy(e_learning.model)

            e_learning.t += 1
            total_timesteps += 1
            step_i += 1
            if done:
                path = {"observations": np.array(observations),
                        "policies":  np.array(policies),
                        "mus": np.array(mus),
                        "logvars": np.array(logvars),
                        "probs_for_actions": np.array(probs_for_actions),
                        "actions": np.array(actions),
                        "rewards": np.array(rewards),
                        "dones": np.array(dones),
                        "cumulative_returns": get_cummulative_returns(rewards_),
                       }
                paths.append(path)
                break
    return paths
