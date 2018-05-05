import time
from itertools import count
from collections import OrderedDict
import gym
import torch
import numpy as np

from trpo.agent import TRPOAgent
from trpo.rollout import rollout, rollout_with_e_learning
from trpo.update import update_step
from trpo.loss import get_discrete_entropy, get_normal_entropy
from tensorboardX import SummaryWriter
from helpers.create_empty_directory import create_empty_directory
from helpers.utils import set_seed


env = gym.make("CartPole-v0")
observation_shape = env.observation_space.shape
n_actions = env.action_space.n
agent = TRPOAgent(observation_shape, n_actions=n_actions)

# action_shape = env.action_space.shape
# agent = TRPOAgent(observation_shape, action_shape=action_shape)


def run_trpo(env, agent,
             max_steps=10,
             max_kl = 0.01,
             num_episodes_per_rollout=10,
             seed=None,
             print_flag=False,
             log_dir='logs/trpo_logs'):
    if seed is not None:
        set_seed(seed, env)

    start_time = time.time()
    numeptotal = 0
    env.reset()

    create_empty_directory(log_dir)
    tensorboard_directory = log_dir + '/tensorboard_logs'
    writer = SummaryWriter(tensorboard_directory)

    optimizer = torch.optim.Adam(agent.values.parameters(), lr=1e-4)
    rewards_per_episode = []
    for i in range(max_steps):

        if print_flag:
            print("\n********** Iteration %i ************" % i)
            print("Rollout")
            paths = rollout(env, agent, num_episodes=num_episodes_per_rollout)
            print("Made rollout")
        else:
            paths = rollout(env, agent, num_episodes=num_episodes_per_rollout)

        # Updating policy.
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        returns = np.concatenate([path["cumulative_returns"] for path in paths])
        probs_for_actions = np.concatenate([path["probs_for_actions"] for path in paths])
        old_policies = np.concatenate([path["policies"] for path in paths])
        old_mu = np.concatenate([path["mus"] for path in paths])
        old_logvar = np.concatenate([path["logvars"] for path in paths])

        if agent.discrete_type:
            loss, kl = update_step(agent, optimizer, observations, actions,
                                   returns, probs_for_actions, max_kl, old_policies=old_policies)
        else:
            loss, kl = update_step(agent, optimizer, observations, actions,
                                   returns, probs_for_actions, max_kl, old_mu=old_mu, old_logvar=old_logvar)
        # Report current progress
        episode_rewards = np.array([path["rewards"].sum() for path in paths])
        rewards_per_episode.append(episode_rewards.mean())

        if torch.cuda.is_available():
            kl = kl.cpu().data.numpy()
            if agent.discrete_type:
                entropy = get_discrete_entropy(agent, observations).cpu().data.numpy()
            else:
                entropy = get_normal_entropy(agent, observations).cpu().data.numpy()
            surrogate_loss = loss.cpu().data.numpy()
        else:
            kl = kl.data.numpy()
            if agent.discrete_type:
                entropy = get_discrete_entropy(agent, observations).data.numpy()
            else:
                entropy = get_normal_entropy(agent, observations).data.numpy()
            surrogate_loss = loss.data.numpy()

        stats = OrderedDict()
        numeptotal += len(episode_rewards)
        stats["Total number of episodes"] = numeptotal
        stats["Average sum of rewards per episode"] = episode_rewards.mean()
        stats["Std of rewards per episode"] = episode_rewards.std()
        stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.)
        stats["KL between old and new distribution"] = kl
        stats["Entropy"] = entropy
        stats["Surrogate loss"] = surrogate_loss


        if print_flag:
            for k, v in stats.items():
                print(k + ": " + " " * (40 - len(k)) + str(v))

        writer.add_scalar('rewards_per_episode',
                          stats["Average sum of rewards per episode"], i)

        writer.add_scalar('std_rewards_per_episode', stats["Std of rewards per episode"], i)
        writer.add_scalar('entropy', stats["Entropy"], i)
        writer.add_scalar('surrogate_loss', stats["Surrogate loss"], i)


    all_episode_rewards = np.array(rewards_per_episode)
    return all_episode_rewards


def run_trpo_with_e_learning(env, agent, e_learning,
                             max_steps=10,
                             max_kl = 0.01,
                             num_episodes_per_rollout=10,
                             seed=None,
                             print_flag=False,
                             log_dir='logs/trpo_logs'):
    if seed is not None:
        set_seed(seed, env)

    start_time = time.time()
    numeptotal = 0
    env.reset()

    create_empty_directory(log_dir)
    tensorboard_directory = log_dir + '/tensorboard_logs'
    writer = SummaryWriter(tensorboard_directory)

    optimizer = torch.optim.Adam(agent.values.parameters(), lr=1e-4)
    rewards_per_episode = []
    for i in range(max_steps):

        if print_flag:
            print("\n********** Iteration %i ************" % i)
            print("Rollout")
            paths = rollout_with_e_learning(env, agent, e_learning,
                                            num_episodes=num_episodes_per_rollout)
            print("Made rollout")
        else:
            paths = rollout_with_e_learning(env, agent, e_learning,
                                            num_episodes=num_episodes_per_rollout)

        # Updating policy.
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        returns = np.concatenate([path["cumulative_returns"] for path in paths])
        probs_for_actions = np.concatenate([path["probs_for_actions"] for path in paths])
        old_policies = np.concatenate([path["policies"] for path in paths])
        old_mu = np.concatenate([path["mus"] for path in paths])
        old_logvar = np.concatenate([path["logvars"] for path in paths])

        if agent.discrete_type:
            loss, kl = update_step(agent, optimizer, observations, actions,
                                   returns, probs_for_actions, max_kl, old_policies=old_policies)
        else:
            loss, kl = update_step(agent, optimizer, observations, actions,
                                   returns, probs_for_actions, max_kl, old_mu=old_mu, old_logvar=old_logvar)
        # Report current progress
        episode_rewards = np.array([path["rewards"].sum() for path in paths])
        rewards_per_episode.append(episode_rewards.mean())

        if torch.cuda.is_available():
            kl = kl.cpu().data.numpy()
            if agent.discrete_type:
                entropy = get_discrete_entropy(agent, observations).cpu().data.numpy()
            else:
                entropy = get_normal_entropy(agent, observations).cpu().data.numpy()
            surrogate_loss = loss.cpu().data.numpy()
        else:
            kl = kl.data.numpy()
            if agent.discrete_type:
                entropy = get_discrete_entropy(agent, observations).data.numpy()
            else:
                entropy = get_normal_entropy(agent, observations).data.numpy()
            surrogate_loss = loss.data.numpy()

        stats = OrderedDict()
        numeptotal += len(episode_rewards)
        stats["Total number of episodes"] = numeptotal
        stats["Average sum of rewards per episode"] = episode_rewards.mean()
        stats["Std of rewards per episode"] = episode_rewards.std()
        stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.)
        stats["KL between old and new distribution"] = kl
        stats["Entropy"] = entropy
        stats["Surrogate loss"] = surrogate_loss

        if print_flag:
            for k, v in stats.items():
                print(k + ": " + " " * (40 - len(k)) + str(v))

        writer.add_scalar('rewards_per_episode',
                          stats["Average sum of rewards per episode"], i)

        writer.add_scalar('std_rewards_per_episode', stats["Std of rewards per episode"], i)
        writer.add_scalar('entropy', stats["Entropy"], i)
        writer.add_scalar('surrogate_loss', stats["Surrogate loss"], i)


    all_episode_rewards = np.array(rewards_per_episode)
    return all_episode_rewards