import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import copy
from helpers.replay_buffer import ReplayBuffer, CountBasedReplayBuffer
from helpers.shedules import LinearSchedule
from helpers.plots import plot_q_func_and_visitations,plot_q_func_and_visitations_and_policy
from helpers.create_empty_directory import create_empty_directory

from tabular_environments.chain_environment import SimpleChain

batch_size = 32


def convert_to_var(arr, astype='float32', add_dim=False):
    if add_dim:
        v = Variable(torch.from_numpy(np.array([arr]).astype(astype)))
    else:
        v = Variable(torch.from_numpy(arr.astype(astype)))
    return v


class DQNnet(nn.Module):
    def __init__(self, num_actions, input_dim, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
                            nn.Linear(input_dim, hidden_size),
                            nn.ReLU(),
                            #nn.Dropout(p=0.1),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            #nn.Dropout(p=0.1),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            #nn.Dropout(p=0.1),
                            nn.Linear(hidden_size, num_actions))

    def forward(self, x):
        out = self.net(x)
        return out


def optimize_dqn_loss(optimizer, model, target_model, batch, gamma,
                      target_type='standard', tau=None, double_dqn=True,
                     ):
    states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = batch
    states_batch_var = convert_to_var(states_batch)
    actions_batch_var = convert_to_var(actions_batch[:, np.newaxis], astype='int64')
    rewards_batch_var = convert_to_var(rewards_batch)
    next_states_batch_var = convert_to_var(next_states_batch)
    dones_batch_var = convert_to_var(dones_batch)

    q_values = model.forward(states_batch_var).gather(1, actions_batch_var)

    if target_type == 'standard_q_learning':
        next_q_values = target_model.forward(next_states_batch_var).detach()
        best_next_q_values = next_q_values.max(dim=1)[0]
        best_next_q_values[dones_batch_var.byte()] = 0
        q_values_targets = rewards_batch_var + gamma * best_next_q_values

    elif target_type == 'double_q_learning':
        all_next_q_values = target_model.forward(next_states_batch_var).detach()
        argmax = torch.max(model.forward(next_states_batch_var), dim=1)[1]
        best_next_q_values = all_next_q_values.gather(1, argmax.view((-1, 1)))[:,0]
        best_next_q_values[dones_batch_var.byte()] = 0
        q_values_targets = rewards_batch_var + gamma * best_next_q_values

    elif target_type == 'soft_q_learning' and tau is not None:
        next_q_values = target_model.forward(next_states_batch_var).detach()
        next_q_values = next_q_values.data.numpy() / tau
        num_actions = next_q_values.shape[1]
        next_q_values = np.logaddexp(next_q_values[:,0], next_q_values[:,1]) - np.log(num_actions)
        next_q_values[dones_batch] = 0
        q_values_targets = rewards_batch + gamma * next_q_values
        q_values_targets = convert_to_var(q_values_targets)

    mse_loss_func = nn.MSELoss()
    loss = mse_loss_func(q_values, q_values_targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data[0]


def eval_agent(env, model, agent_type='simple_dqn'):
    episode_total_reward = 0
    num_actions = env.action_space.n
    state = env.reset()
    log_file = 'logs/last_eval_q_func.csv'
    with open(log_file, 'w'):
        pass
    while True:
        action = epsilon_greedy_act(num_actions, state, model, 0, log_file=log_file)
        next_state, rew, done, _ = env.step(action)
        state = next_state
        episode_total_reward += rew
        if done:
            break
    env.reset()
    return episode_total_reward


def write_tensorboard_logs(lcl):
    lcl['writer'].add_scalar('dqn/metrics/loss', lcl['loss'], lcl['t'])
    all_states_var = convert_to_var(lcl['env'].get_all_states())

    plot_q_func = lcl['act_type'] in ['epsilon_greedy', 'ucb', 'soft_policy', 'soft_policies_mixtures']
    plot_q_func_policy = lcl['act_type'] in ['soft_policy', 'soft_policies_mixtures']

    if plot_q_func:
        all_q_values = lcl['model'].forward(all_states_var)
        probs = F.softmax(all_q_values / lcl['tau_t'], dim=1)
        all_q_values = all_q_values.data.numpy()
        probs = probs.data.numpy()
        for i in range(lcl['n_all_states']):
            if (2 <= i < lcl['n_all_states'] - 2) and lcl['n_all_states'] >= 10:
                continue
            else:
                lcl['writer'].add_scalars('dqn/q_values/state_{}'.format(i + 1), {'action_right': all_q_values[i][1],
                                                                                   'action_left': all_q_values[i][0]},
                                          lcl['t'])
                if plot_q_func_policy:
                    lcl['writer'].add_scalars('dqn/policy_probs/state_{}'.format(i + 1), {'action_right': probs[i][1],
                                                                                          'action_left': probs[i][0]},
                                              lcl['t'])

    plot_exploration_q_func = lcl['act_type'] in ['exploration_epsilon_greedy',
                                                  'exploration_ucb',
                                                  'exploration_soft_policy', 'soft_policies_mixtures']
    plot_exploration_q_func_probs = lcl['act_type'] in ['exploration_soft_policy', 'soft_policies_mixtures']

    if plot_exploration_q_func:
        all_exploration_q_values = lcl['exploration_model'].forward(all_states_var)
        exploration_probs = F.softmax(all_exploration_q_values / lcl['tau_t'], dim=1)
        all_exploration_q_values = all_exploration_q_values.data.numpy()
        exploration_probs = exploration_probs.data.numpy()
        for i in range(lcl['n_all_states']):
            if (2 <= i < lcl['n_all_states'] - 2) and lcl['n_all_states'] >= 10:
                continue
            else:
                lcl['writer'].add_scalars('dqn/exploration_ q_values/state_{}'.format(i + 1),
                                          {'action_right': all_exploration_q_values[i][1],
                                           'action_left': all_exploration_q_values[i][0]},
                                          lcl['t'])
                if plot_exploration_q_func_probs:
                    lcl['writer'].add_scalars('dqn/exploration_policy_probs/state_{}'.format(i + 1),
                                              {'action_right': exploration_probs[i][1],
                                               'action_left': exploration_probs[i][0]},
                                              lcl['t'])


    lcl['writer'].add_scalar('dqn/metrics/count_good_reward', lcl['count_good_rewards'], lcl['t'])
    lcl['writer'].add_scalar('dqn/metrics/eps_t', lcl['eps_t'], lcl['t'])
    lcl['writer'].add_scalar('dqn/metrics/tau_t', lcl['tau_t'], lcl['t'])
    lcl['writer'].add_scalar('dqn/metrics/rew_addition', lcl['rew_addition'], lcl['t'])
    lcl['writer'].add_scalar('dqn/metrics/entropy', lcl['entropy'], lcl['t'])
    if lcl['done']:
        lcl['writer'].add_scalar('dqn/metrics/sum_rewards_per_episode', lcl['sum_rewards_per_episode'][-1], lcl['num_episodes'])

    if lcl['t'] > lcl['learning_starts_in_steps'] and lcl['t'] % lcl['train_freq_in_steps'] == 0:
        lcl['writer'].add_scalar('dqn/metrics/rewards_batch', np.sum(lcl['batch'][2]), lcl['t'])

    if lcl['num_episodes'] % lcl['plot_freq'] == 0 and lcl['done']:
        if plot_q_func and not plot_q_func_policy:
            plot_q_func_and_visitations( lcl['episode_visitations'],
                                        lcl['state_action_count'], all_q_values,
                                        lcl['num_episodes'], lcl['t'], lcl['images_directory'])
        if plot_q_func and plot_q_func_policy:
            plot_q_func_and_visitations_and_policy(lcl['episode_visitations'],
                                                   lcl['state_action_count'], all_q_values, probs,
                                                   lcl['num_episodes'], lcl['t'], lcl['images_directory'])

        if plot_exploration_q_func and plot_exploration_q_func_probs:
            plot_q_func_and_visitations_and_policy(lcl['state_action_count'].sum(axis=1), lcl['episode_visitations'],
                                                   lcl['state_action_count'], all_exploration_q_values, exploration_probs,
                                                   lcl['num_episodes'], lcl['t'], lcl['images_directory'])

        if plot_exploration_q_func and not plot_exploration_q_func_probs:
            plot_q_func_and_visitations( lcl['episode_visitations'],
                                        lcl['state_action_count'], all_exploration_q_values,
                                        lcl['num_episodes'], lcl['t'], lcl['images_directory'])


def check_that_solved(lcl):
    if lcl['done'] and lcl['eval_freq'] is not None and lcl['num_episodes'] % lcl['eval_freq'] == 0:
        test_episode_reward = eval_agent(lcl['env'], lcl['model'])
        if test_episode_reward == 10 and lcl['count_good_rewards'] > 0:
            print('Successfully solved environment in {} episodes'.format(lcl['num_episodes']))
            return True


def check_that_done(lcl):
    if lcl['done']:
        print('Episode:', lcl['num_episodes'], lcl['sum_rewards_per_episode'][-1])
        lcl['episode_visitations'] = np.zeros(lcl['dim_states'])
        lcl['num_episodes'] += 1
        lcl['sum_rewards_per_episode'].append(0)
        lcl['list_rewards_per_episode'].append([])
        lcl['state'] = lcl['env'].reset()


def count_rew_addition(state_action_count, state_id, next_state_id, action, type):
    eps = np.finfo(float).eps
    if type == 'count_based_state':
        add = 1 / (np.sqrt(state_action_count[state_id].sum()) + 1)

    elif type == 'count_based_state_action':
        add = 1 / (np.sqrt(state_action_count[state_id][action]) + 1)

    elif type == 'count_based_next_state':
        add = 1 / (np.sqrt(state_action_count[next_state_id].sum()) + 1)

    elif type == 'count_based_next_state_action':
        add = 1 / (np.sqrt(state_action_count[next_state_id][action]) + 1)
    else:
        add = 0

    return add


def epsilon_greedy_act(num_actions, state, model, eps_t, ucb=None, log_file=None):
    state_var = convert_to_var(state, add_dim=True)
    q_values = model.forward(state_var).data.numpy()[0]

    if ucb is not None:
        q_values += ucb

    if np.random.rand() < eps_t:
        action = np.random.randint(num_actions)
        flag='random'
    else:
        action = q_values.argmax()
        flag='argmax'
    if log_file is not None:
        s = str(action) + ',' + ','.join([str(q) for q in q_values]) + '\n'
        with open(log_file, 'a') as f:
            f.write(s)
    return action, flag


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def soft_policy_act(num_actions, state, model, tau_t):
    state_var = convert_to_var(state, add_dim=True)
    q_values = model.forward(state_var)#.data.numpy()[0]
    policy = F.softmax(q_values / tau_t, dim=1).data.numpy()[0]
    log_policy = F.log_softmax(q_values / tau_t, dim=1).data.numpy()[0]
    action = np.random.choice(num_actions, p=policy)
    entropy = -1 * (policy * log_policy).sum()
    return action, entropy


def soft_policies_mixture_act(num_actions, state,
                              model, exploration_model, tau_t,
                              exploration_coef = 0.5):
    state_var = convert_to_var(state, add_dim=True)

    q_values = model.forward(state_var)#.data.numpy()[0]
    policy = F.softmax(q_values / tau_t, dim=1).data.numpy()[0]
    log_policy = F.log_softmax(q_values / tau_t, dim=1).data.numpy()[0]
    entropy = -1 * (policy * log_policy).sum()

    exploration_q_values = exploration_model.forward(state_var)
    exploration_policy = F.softmax(exploration_q_values / tau_t, dim=1).data.numpy()[0]
    exploration_log_policy = F.log_softmax(exploration_q_values / tau_t, dim=1).data.numpy()[0]
    exploration_entropy = -1 * (exploration_policy * exploration_log_policy).sum()

    mixture = (1 - exploration_coef) * policy + exploration_coef * exploration_policy
    mixture_entropy = -1 * (mixture * np.log(mixture)).sum()

    action = np.random.choice(num_actions, p=mixture)
    return action, mixture_entropy


def pretrain(model, all_states, num_actions,
             max_steps, eps, writer):
    n_states = all_states.shape[0]
    target = convert_to_var(np.ones((n_states,num_actions)) / num_actions)
    all_states = convert_to_var(all_states)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    mse = nn.MSELoss()
    for t in range(max_steps):
        q = model.forward(all_states)

        loss = mse(q, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(n_states):
            if (2 <= i < n_states - 2) and n_states >= 10:
                continue
            else:
                writer.add_scalars('dqn/q_values/state_{}'.format(i + 1), {'action_right': q[i][1],
                                                                           'action_left': q[i][0]}, t)
        if loss.data.numpy()[0] < eps:
            return t

    return max_steps


def train(env,
          eps_params=None,
          tau_params=None,
          alpha_params=None,
          gamma=0.99,
          max_steps=100,
          learning_starts_in_steps=100,
          train_freq_in_steps=1,
          update_freq_in_steps=10,
          plot_freq=1,
          eval_freq=5,
          seed=None,
          log_dir='logs',
          write_logs=None,
          count_based_exploration_type='state_action',
          act_type='epsilon_greedy',
          target_type='standard',
          reward_shaping_type=None,
          do_pretraining=False
          ):

    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)

    num_actions = env.action_space.n
    dim_states = env.observation_space.shape[0]
    n_all_states = env.get_all_states().shape[0]

    create_empty_directory(log_dir)
    tensorboard_directory = log_dir + '/tensorboard_logs'
    images_directory = log_dir+'/images_logs'
    descr_file_name = log_dir + '/parameters.csv'

    with open(descr_file_name,'w') as file:
        file.write('dim_states: {}\n'.format(dim_states))
        file.write('seed: {}\n'.format(seed))
        file.write('action selection type: {}\n'.format(act_type))
        file.write('target q-function type: {}\n'.format(target_type))
        file.write('reward addition type: {}\n'.format(reward_shaping_type))
        file.write('Without epsilon: {}\n'.format(True if eps_params is None else False))
        file.write('count-based exploration type (rewards in exploration model): {}\n'.format(count_based_exploration_type))

    # create empty directories for writing logs
    create_empty_directory(images_directory)
    create_empty_directory(tensorboard_directory)
    writer = SummaryWriter(tensorboard_directory)

    # define models
    model = DQNnet(num_actions, dim_states)
    if do_pretraining:
        pretrain(model, env.get_all_states(), num_actions,
                 eps=1e-5, max_steps=int(1e3), writer=writer)
    target_model = copy.deepcopy(model)
    exploration_model = DQNnet(num_actions, dim_states)
    exploration_target_model = copy.deepcopy(exploration_model)

    # define optimizator
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    exploration_optimizer = torch.optim.Adam(exploration_model.parameters(), lr=1e-5)

    # define shedule of epsilon in epsilon-greedy exploration
    if eps_params is not None:
        schedule_timesteps = int(eps_params['exploration_fraction'] * max_steps)
        eps_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                     initial_p=1.0,
                                     final_p=eps_params['exploration_final_eps'])
    else:
        eps_shedule = None

    # define shedule of tau
    if tau_params is not None:
        schedule_timesteps = int(tau_params['fraction'] * max_steps)
        tau_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                     initial_p=1.0,
                                     final_p=tau_params['final_tau'])
    else:
        tau_shedule = None

    if alpha_params is not None:
        schedule_timesteps = int(alpha_params['fraction'] * max_steps)
        alpha_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                       initial_p=alpha_params['initial_alpha'],
                                       final_p=alpha_params['final_alpha'])
    else:
        alpha_shedule = None

    # create replay buffers
    replay_buffer = ReplayBuffer(1000, seed=seed)
    exploration_replay_buffer = CountBasedReplayBuffer(10000,
                                                       count_based_exploration_type,
                                                       env.convert_state_to_id_func(),
                                                       seed=seed)

    num_episodes = 0
    sum_rewards_per_episode = [0]
    list_rewards_per_episode = [[]]
    state_action_count = np.zeros((n_all_states, num_actions))
    count_good_rewards = 0
    episode_visitations = np.zeros((dim_states, num_actions))
    episode_history = []
    state = env.reset()
    break_flag=False

    for t in range(max_steps):
        eps_t = eps_shedule.value(t) if eps_shedule else 0
        tau_t = tau_shedule.value(t) if tau_shedule else 0
        alpha_t = alpha_shedule.value(t) if alpha_shedule else 1

        if act_type == 'epsilon_greedy':
            action, flag = epsilon_greedy_act(num_actions, state, model, eps_t)
            entropy = 0
            episode_history.append((env.convert_state_to_id(state), action, flag))

        elif act_type == 'ucb-1':
            s_id = env.convert_state_to_id(state)
            n_s = max(1, state_action_count[s_id].sum())
            n_sa = np.maximum(1, state_action_count[s_id])
            ucb = np.sqrt(2 * np.log(n_s) / n_sa)
            action, flag = epsilon_greedy_act(num_actions, state, model, eps_t, ucb=alpha_t * ucb)
            entropy = 0

        elif act_type == 'ucb-2':
            s_id = env.convert_state_to_id(state)
            ucb = 1 / np.sqrt(1 + state_action_count[s_id])
            action, flag = epsilon_greedy_act(num_actions, state, model, eps_t, ucb=alpha_t * ucb)
            entropy = 0

        elif act_type == 'exploration_epsilon_greedy':
            action, flag = epsilon_greedy_act(num_actions, state, exploration_model, eps_t)
            entropy = 0
            episode_history.append((env.convert_state_to_id(state), action, flag))
        elif act_type == 'soft_policy':
            action, entropy = soft_policy_act(num_actions, state, model, tau_t)
        elif act_type == 'exploration_soft_policy':
            action, entropy = soft_policy_act(num_actions, state, exploration_model, tau_t)
        elif act_type == 'soft_policies_mixtures':
            action, entropy = soft_policies_mixture_act(num_actions, state, model, exploration_model, tau_t)

        next_state, rew, done, _ = env.step(action)
        rew_addition = count_rew_addition(state_action_count,
                                          env.convert_state_to_id(state),
                                          env.convert_state_to_id(next_state),
                                          action, reward_shaping_type)

        replay_buffer.add(state, action, rew + rew_addition, next_state, done)
        exploration_replay_buffer.add(state, action, next_state, done)

        state_action_count[env.convert_state_to_id(state)][action] += 1
        episode_visitations[env.convert_state_to_id(state)][action] += 1

        if rew == 1:
            count_good_rewards += 1
        sum_rewards_per_episode[-1] += rew
        list_rewards_per_episode[-1].append(rew)

        state = next_state

        if t > learning_starts_in_steps and t % train_freq_in_steps == 0:
            batch = replay_buffer.sample(batch_size)
            loss = optimize_dqn_loss(optimizer, model, target_model, batch, gamma,
                                     target_type=target_type, tau=tau_t)

            exploration_batch = exploration_replay_buffer.sample(batch_size, state_action_count)
            exploration_loss = optimize_dqn_loss(exploration_optimizer, exploration_model,
                                                 exploration_target_model, exploration_batch, gamma,
                                                 target_type=target_type, tau=tau_t)
        else:
            loss = 0
            exploration_loss = 0

        if t > learning_starts_in_steps and t % update_freq_in_steps == 0:
            target_model = copy.deepcopy(model)
            exploration_target_model = copy.deepcopy(exploration_model)

        if write_logs:
            write_tensorboard_logs(locals())

        if done:
            print('Episode:', num_episodes, sum_rewards_per_episode[-1])
            if np.sum(sum_rewards_per_episode[-100:]) == 100*10:
                break
            episode_history=[]
            episode_visitations = np.zeros((dim_states, num_actions))
            num_episodes += 1
            sum_rewards_per_episode.append(0)
            list_rewards_per_episode.append([])
            state = env.reset()

    return state_action_count, num_episodes


if __name__ == "__main__":
    eps_params = {'exploration_fraction': 0.25,
                  'exploration_final_eps': 0.001}

    common_params = dict(gamma=0.99, write_logs=True, log_dir='logs/simple_experiment',
                         plot_freq=10, target_type='double_q_learning')

    params = dict(eps_params=eps_params,
                  act_type='epsilon_greedy', reward_shaping_type=None)

    dim=5
    seed=12
    env = SimpleChain(int(dim))
    _, num_episodes = train(env,
                            seed=seed,
                            learning_starts_in_steps=(dim + 9) * 3,
                            max_steps=2000 * (dim + 9),
                            train_freq_in_steps=10,
                            update_freq_in_steps=60,
                            **common_params, **params)

    print(num_episodes)