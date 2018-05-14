import numpy as np
import copy
import torch
from tensorboardX import SummaryWriter

from helpers.create_empty_directory import create_empty_directory
from qlearning.pretrain import pretrain
from qlearning.act import epsilon_greedy_act, lll_epsilon_greedy_act
from qlearning.q_loss import dqn_loss, sarsa_loss
from qlearning.write_logs import write_tensorboard_tabular_logs, write_tensorboard_logs
from helpers.replay_buffer import ReplayBuffer, CountBasedReplayBuffer
from helpers.shedules import LinearSchedule
from helpers.utils import set_seed
from helpers.convert_to_var_foo import convert_to_var
from helpers.plots import plot_visitations_with_e_values, plot_corr_hist

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


def eval_greedy_agent(env, model):
    episode_total_reward = 0
    num_actions = env.action_space.n
    state = env.reset()
    while True:
        action = epsilon_greedy_act(num_actions, state, model, 0)
        next_state, rew, done, _ = env.step(action)
        state = next_state
        episode_total_reward += rew
        if done:
            break
    env.reset()
    return episode_total_reward


def train_tabular(env, model,
                  eps_params=None,
                  alpha_params=None,
                  lr=1e-5,
                  replay_buffer_size=1000,
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
                  do_pretraining=False,
                  print_freq=1,
                  batch_size=32,

                ):

    if seed:
        set_seed(seed, env)

    num_actions = env.action_space.n
    dim_states = env.observation_space.shape[0]
    n_all_states = env.get_all_states().shape[0]

    create_empty_directory(log_dir)
    tensorboard_directory = log_dir + '/tensorboard_logs'
    images_directory = log_dir + '/images_logs'
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

    # create a summary writer
    writer = SummaryWriter(tensorboard_directory)

    # define models
    if do_pretraining:
        pretrain(model, env.get_all_states(), num_actions,
                 eps=1e-6, max_steps=int(1e5), writer=writer)
    target_model = copy.deepcopy(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define shedule of epsilon in epsilon-greedy exploration
    if eps_params is not None:
        schedule_timesteps = int(eps_params['exploration_fraction'] * max_steps)
        eps_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                     initial_p=1.0,
                                     final_p=eps_params['exploration_final_eps'])
    else:
        eps_shedule = None

    if alpha_params is not None:
        schedule_timesteps = int(alpha_params['fraction'] * max_steps)
        alpha_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                       initial_p=alpha_params['initial_alpha'],
                                       final_p=alpha_params['final_alpha'])
    else:
        alpha_shedule = None

    # create replay buffers
    replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)

    num_episodes = 0
    sum_rewards_per_episode = [0]
    list_rewards_per_episode = [[]]
    state_action_count = np.zeros((n_all_states, num_actions))
    count_good_rewards = 0
    episode_visitations = np.zeros((dim_states, num_actions))
    episode_history = []
    state = env.reset()
    state_history = []
    state_ids = []
    count_history = []
    break_flag=False

    for t in range(max_steps):
        eps_t = eps_shedule.value(t) if eps_shedule else 0
        alpha_t = alpha_shedule.value(t) if alpha_shedule else 1

        if act_type == 'epsilon_greedy':
            action = epsilon_greedy_act(num_actions, state, model, eps_t)
            entropy = 0

        elif act_type == 'ucb-1':
            s_id = env.convert_state_to_id(state)
            n_s = max(1, state_action_count[s_id].sum())
            n_sa = np.maximum(1, state_action_count[s_id])
            ucb = np.sqrt(2 * np.log(n_s) / n_sa)
            action = epsilon_greedy_act(num_actions, state, model, eps_t, ucb=alpha_t * ucb)
            entropy = 0

        elif act_type == 'ucb-2':
            s_id = env.convert_state_to_id(state)
            ucb = 1 / np.sqrt(1 + state_action_count[s_id])
            action = epsilon_greedy_act(num_actions, state, model, eps_t, ucb=alpha_t * ucb)
            entropy = 0
        next_state, rew, done, _ = env.step(action)
        rew_addition = count_rew_addition(state_action_count,
                                          env.convert_state_to_id(state),
                                          env.convert_state_to_id(next_state),
                                          action, reward_shaping_type)

        replay_buffer.add(state, action, rew + rew_addition, done, next_state, -1)

        state_action_count[env.convert_state_to_id(state)][action] += 1
        episode_visitations[env.convert_state_to_id(state)][action] += 1

        state_history.append(state)
        count_history.append(state_action_count)
        state_ids.append(env.convert_state_to_id(state))

        if rew == 1:
            count_good_rewards += 1
        sum_rewards_per_episode[-1] += rew
        list_rewards_per_episode[-1].append(rew)

        state = next_state

        if t > learning_starts_in_steps and t % train_freq_in_steps == 0:
            batch = replay_buffer.sample(batch_size)
            loss = dqn_loss(optimizer, model, target_model, batch, gamma,
                                     target_type=target_type)
        else:
            loss = 0
            exploration_loss = 0

        if t > learning_starts_in_steps and t % update_freq_in_steps == 0:
            target_model = copy.deepcopy(model)

        if write_logs:
            write_tensorboard_tabular_logs(locals())

        if done:
            if print_freq is not None:
                print('Episode:', num_episodes, sum_rewards_per_episode[-1])
            if np.sum(sum_rewards_per_episode[-100:]) == 100*10:
                break

            episode_history=[]
            episode_visitations = np.zeros((dim_states, num_actions))
            num_episodes += 1
            sum_rewards_per_episode.append(0)
            list_rewards_per_episode.append([])
            state = env.reset()

    return np.array(state_history), np.array(count_history), np.array(state_ids)

    if done:
        return sum_rewards_per_episode[-2], num_episodes
    else:
        return sum_rewards_per_episode[-1], num_episodes


def train_tabular_with_e_learning(env, model, e_model,
                  e_lr=1e-4,
                  gamma_E=0.9,
                  eps_params=None,
                  alpha_params=None,
                  lr=1e-5,
                  replay_buffer_size=1000,
                  gamma=0.99,
                  max_steps=100,
                  max_num_episodes=1,
                  learning_starts_in_steps=100,
                  train_freq_in_steps=1,
                  e_train_freq_in_steps=1,
                  update_freq_in_steps=10,
                  e_update_freq_in_steps=10,
                  plot_freq=1,
                  eval_freq=5,
                  seed=None,
                  log_dir='logs',
                  write_logs=None,
                  count_based_exploration_type='state_action',
                  act_type='epsilon_greedy',
                  target_type='standard',
                  reward_shaping_type=None,
                  do_pretraining=False,
                  print_freq=1,
                  batch_size=32
                ):
    if seed:
        set_seed(seed, env)

    num_actions = env.action_space.n
    dim_states = env.observation_space.shape[0]
    n_all_states = env.get_all_states().shape[0]

    create_empty_directory(log_dir)
    tensorboard_directory = log_dir + '/tensorboard_logs'
    images_directory = log_dir + '/images_logs'
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

    # create a summary writer
    writer = SummaryWriter(tensorboard_directory)

    # define models
    if do_pretraining:
        pretrain(model, env.get_all_states(), num_actions,
                 eps=1e-6, max_steps=int(1e5), writer=writer)
    target_model = copy.deepcopy(model)
    target_e_model = copy.deepcopy(e_model)
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    e_optimizer = torch.optim.Adam(e_model.parameters(), lr=e_lr)

    # define shedule of epsilon in epsilon-greedy exploration
    if eps_params is not None:
        schedule_timesteps = int(eps_params['exploration_fraction'] * max_steps)
        eps_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                     initial_p=1.0,
                                     final_p=eps_params['exploration_final_eps'])
    else:
        eps_shedule = None

    if alpha_params is not None:
        schedule_timesteps = int(alpha_params['fraction'] * max_steps)
        alpha_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                       initial_p=alpha_params['initial_alpha'],
                                       final_p=alpha_params['final_alpha'])
    else:
        alpha_shedule = None

    # create replay buffers
    replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)

    num_episodes = 0
    sum_rewards_per_episode = [0]
    list_rewards_per_episode = [[]]
    state_action_count = np.zeros((n_all_states, num_actions))
    count_good_rewards = 0
    episode_visitations = np.zeros((dim_states, num_actions))
    episode_history = []
    state = env.reset()
    break_flag=False

    corr_coefs = []

    t = 0
    for i_episode in range(max_num_episodes):
        state = env.reset()

        eps_t = eps_shedule.value(t) if eps_shedule else 0
        alpha_t = alpha_shedule.value(t) if alpha_shedule else 1

        if act_type == 'epsilon_greedy':
            action = epsilon_greedy_act(num_actions, state, model, eps_t)
            entropy = 0
            episode_history.append((env.convert_state_to_id(state), action))

        elif act_type == 'ucb-1':
            s_id = env.convert_state_to_id(state)
            n_s = max(1, state_action_count[s_id].sum())
            n_sa = np.maximum(1, state_action_count[s_id])
            ucb = np.sqrt(2 * np.log(n_s) / n_sa)
            action = epsilon_greedy_act(num_actions, state, model, eps_t, ucb=alpha_t * ucb)
            entropy = 0

        elif act_type == 'ucb-2':
            s_id = env.convert_state_to_id(state)
            ucb = 1 / np.sqrt(1 + state_action_count[s_id])
            action, flag = epsilon_greedy_act(num_actions, state, model, eps_t, ucb=alpha_t * ucb)
            entropy = 0

        while True:
            next_state, rew, done, _ = env.step(action)
            rew_addition = count_rew_addition(state_action_count,
                                              env.convert_state_to_id(state),
                                              env.convert_state_to_id(next_state),
                                              action, reward_shaping_type)

            eps_t = eps_shedule.value(t) if eps_shedule else 0
            alpha_t = alpha_shedule.value(t) if alpha_shedule else 1
            if act_type == 'epsilon_greedy':
                next_action = epsilon_greedy_act(num_actions, state, model, eps_t)
                entropy = 0
                episode_history.append((env.convert_state_to_id(state), action))
            elif act_type == 'ucb-1':
                s_id = env.convert_state_to_id(state)
                n_s = max(1, state_action_count[s_id].sum())
                n_sa = np.maximum(1, state_action_count[s_id])
                ucb = np.sqrt(2 * np.log(n_s) / n_sa)
                next_action = epsilon_greedy_act(num_actions, state, model, eps_t, ucb=alpha_t * ucb)
                entropy = 0
            elif act_type == 'ucb-2':
                s_id = env.convert_state_to_id(state)
                ucb = 1 / np.sqrt(1 + state_action_count[s_id])
                next_action = epsilon_greedy_act(num_actions, state, model, eps_t, ucb=alpha_t * ucb)
                entropy = 0
            else:
                print('Unknow action selection type')

            replay_buffer.add(state, action, rew + rew_addition, done, next_state, next_action)

            state_action_count[env.convert_state_to_id(state)][action] += 1
            episode_visitations[env.convert_state_to_id(state)][action] += 1

            if rew == 1:
                count_good_rewards += 1
            sum_rewards_per_episode[-1] += rew
            list_rewards_per_episode[-1].append(rew)

            if t > learning_starts_in_steps:
                batch = [np.array([state]), np.array([action]), np.array([rew]),
                         np.array([done]), np.array([next_state]), np.array([next_action])]
                e_loss = sarsa_loss(e_optimizer, e_model, target_e_model, batch, gamma_E)

            if t > learning_starts_in_steps and t % train_freq_in_steps == 0:
                batch = replay_buffer.sample(batch_size)
                loss = dqn_loss(optimizer, model, target_model, batch, gamma,
                                target_type=target_type)
            else:
                loss = 0

            if t > learning_starts_in_steps and t % e_train_freq_in_steps == 0:
                batch = replay_buffer.sample(batch_size)
                states = batch[0]
                states_ids = []
                for st in states:
                    states_ids.append(env.convert_state_to_id(st))
                states_ids = np.array(states_ids)
                unique, counts = np.unique(states_ids, return_counts=True)
                tmp = [(u,c) for u, c in zip(unique, counts)]
                print(tmp)
                e_loss = sarsa_loss(e_optimizer, e_model, target_e_model, batch,
                                    gamma_E)
            else:
                e_loss = 0

            if t > learning_starts_in_steps and t % update_freq_in_steps == 0:
                target_model = copy.deepcopy(model)

            if t > learning_starts_in_steps and t % e_update_freq_in_steps == 0:
                target_e_model = copy.deepcopy(e_model)

            if done:
                all_states_var = convert_to_var(env.get_all_states())
                if torch.cuda.is_available():
                    e_values = e_model.forward(all_states_var).cpu().data.numpy()
                else:
                    e_values = e_model.forward(all_states_var).data.numpy()
                e_counters = np.log(e_values) / np.log(1-e_lr) + np.log(2)/np.log(1-e_lr)
                plot_visitations_with_e_values(state_action_count, e_values, e_counters,
                                               num_episodes, t, images_directory)
                if t > learning_starts_in_steps:
                    corr_coefs.append(np.corrcoef(state_action_count.flatten(),
                                      e_counters.flatten())[0][1])

            state = next_state
            action = next_action
            t += 1
            if done:
                if print_freq is not None:
                    print('t: {}, Episode: {}, sum: {}, eps: {:.2f}'. \
                          format(t, num_episodes, sum_rewards_per_episode[-1], eps_t))
                episode_history=[]
                episode_visitations = np.zeros((dim_states, num_actions))
                num_episodes += 1
                sum_rewards_per_episode.append(0)
                list_rewards_per_episode.append([])
                state = env.reset()
                break
    torch.save(e_model, 'e_model')
    plot_corr_hist(corr_coefs)
    np.save('corr_coefs', np.array(corr_coefs))
    return sum_rewards_per_episode[-2], num_episodes


def train(env, model,
          eps_params=None,
          lr=1e-5,
          replay_buffer_size=1000,
          gamma=0.99,
          max_steps=100,
          max_num_episodes=1,
          learning_starts_in_steps=100,
          train_freq_in_steps=1,
          update_freq_in_steps=10,
          seed=None,
          log_dir='logs',
          write_logs=None,
          act_type='epsilon_greedy',
          target_type='standard',
          print_freq=1,
          batch_size=32,
          return_states=False,
          ):

    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)

    num_actions = env.action_space.n
    dim_states = env.observation_space.shape[0]

    create_empty_directory(log_dir)
    tensorboard_directory = log_dir + '/tensorboard_logs'
    descr_file_name = log_dir + '/parameters.csv'

    with open(descr_file_name,'w') as file:
        file.write('dim_states: {}\n'.format(dim_states))
        file.write('seed: {}\n'.format(seed))
        file.write('action selection type: {}\n'.format(act_type))
        file.write('target q-function type: {}\n'.format(target_type))
        file.write('Without epsilon: {}\n'.format(True if eps_params is None else False))

    # create empty directories for writing logs
    create_empty_directory(tensorboard_directory)

    # create a summary writer
    writer = SummaryWriter(tensorboard_directory)

    # define models
    target_model = copy.deepcopy(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define shedule of epsilon in epsilon-greedy exploration
    if eps_params is not None:
        schedule_timesteps = int(eps_params['exploration_fraction'] * max_num_episodes)
        print('schedule_timesteps',schedule_timesteps)
        eps_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                     initial_p=1.0,
                                     final_p=eps_params['exploration_final_eps'])
    else:
        eps_shedule = None

    # create replay buffers
    replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)

    num_episodes = 0
    sum_rewards_per_episode = [0]
    list_rewards_per_episode = [[]]
    state_history = []
    count_history = []
    state = env.reset()
    break_flag=False

    for t in range(max_steps):
        eps_t = eps_shedule.value(num_episodes) if eps_shedule else 0

        action = epsilon_greedy_act(num_actions, state, model, eps_t)
        next_state, rew, done, _ = env.step(action)
        replay_buffer.add(state, action, rew, done, next_state, _)

        sum_rewards_per_episode[-1] += rew
        list_rewards_per_episode[-1].append(rew)
        if done:
            print('DONE')
            print(state, next_state)
            print()

        state = next_state

        if t > learning_starts_in_steps and t % train_freq_in_steps == 0:
            batch = replay_buffer.sample(batch_size)
            loss = dqn_loss(optimizer, model, target_model, batch, gamma,
                            target_type=target_type)
        else:
            loss = 0
            exploration_loss = 0

        if t > learning_starts_in_steps and t % update_freq_in_steps == 0:
            target_model = copy.deepcopy(model)

        if write_logs:
            write_tensorboard_logs(locals())

        if done:
            if print_freq is not None:
                print('t: {}, Episode: {}, sum: {}, eps: {:.2f}'.\
                      format(t, num_episodes, sum_rewards_per_episode[-1], eps_t))
            num_episodes += 1
            sum_rewards_per_episode.append(0)

            state = env.reset()

        if len(sum_rewards_per_episode) > max_num_episodes:
            break

    state_history = np.array(state_history)

    if return_states:
        if done:
            return sum_rewards_per_episode[:-1], num_episodes-1, state_history
        else:
            return sum_rewards_per_episode, num_episodes, state_history
    else:
        if done:
            return sum_rewards_per_episode[:-1], num_episodes-1
        else:
            return sum_rewards_per_episode, num_episodes




def train_with_e_learning(env, model, e_model,
                          add_ucb=False,
                          add_bonus=False,
                          beta=1,
                          eps_params=None,
                          e_lr=1e-5,
                          lr=1e-5,
                          replay_buffer_size=1000,
                          gamma=0.99,
                          gamma_E=0.9,
                          max_steps=100,
                          max_num_episodes=1,
                          learning_starts_in_steps=100,
                          train_freq_in_steps=1,
                          e_train_freq_in_steps=1,
                          update_freq_in_steps=10,
                          e_update_freq_in_steps=10,
                          seed=None,
                          log_dir='logs',
                          write_logs=None,
                          act_type='epsilon_greedy',
                          target_type='standard',
                          print_freq=1,
                          batch_size=32,
                          do_pretraining=False,
                          chain_criterion=False,
                          return_states=False,
                          return_trajectory=False,
                          trajectory_path=''
                          ):

    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

    num_actions = env.action_space.n
    dim_states = env.observation_space.shape[0]

    create_empty_directory(log_dir)
    tensorboard_directory = log_dir + '/tensorboard_logs'
    descr_file_name = log_dir + '/parameters.csv'

    with open(descr_file_name,'w') as file:
        file.write('dim_states: {}\n'.format(dim_states))
        file.write('seed: {}\n'.format(seed))
        file.write('action selection type: {}\n'.format(act_type))
        file.write('target q-function type: {}\n'.format(target_type))
        file.write('Without epsilon: {}\n'.format(True if eps_params is None else False))

    # create empty directories for writing logs
    create_empty_directory(tensorboard_directory)
    # create a summary writer
    writer = SummaryWriter(tensorboard_directory)

    # define models
    target_model = copy.deepcopy(model)
    target_e_model = copy.deepcopy(e_model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    e_optimizer = torch.optim.RMSprop(e_model.parameters(), momentum=0.9, lr=e_lr)

    # define shedule of epsilon in epsilon-greedy exploration
    if eps_params is not None:
        schedule_timesteps = int(eps_params['exploration_fraction'] * max_num_episodes)
        print('schedule_timesteps', schedule_timesteps)
        eps_shedule = LinearSchedule(schedule_timesteps=schedule_timesteps,
                                     initial_p=1.0,
                                     final_p=eps_params['exploration_final_eps'])
    else:
        eps_shedule = None

    # create replay buffers
    replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)

    num_episodes = 0
    sum_rewards_per_episode = [0]
    list_rewards_per_episode = [[]]
    state_history = []
    trajectory = []
    state = env.reset()
    break_flag=False

    t = 1
    max_state = 0
    for episode in range(max_num_episodes):
        ucb_work = 0
        episode_trajectory = []
        if len(sum_rewards_per_episode) > max_num_episodes:
            break

        if len(sum_rewards_per_episode) > 10 and t > learning_starts_in_steps  and chain_criterion:
            if np.sum(sum_rewards_per_episode[-11:]) == 10*10:
                break

        eps_t = eps_shedule.value(episode+1) if eps_shedule else 0

        if add_ucb:
            if torch.cuda.is_available():
                e_values = e_model.forward(convert_to_var(state)).cpu().data.numpy()
                q_values = model.forward(convert_to_var(state)).cpu().data.numpy()
            else:
                e_values = e_model.forward(convert_to_var(state)).data.numpy()
                q_values = model.forward(convert_to_var(state)).data.numpy()
            
            cnt = np.log(e_values) / np.log(1-e_lr) + np.log(2) / np.log(1-e_lr)
            ucb = np.sqrt(2 * np.log(t) / cnt)
            ucb *= beta

            if q_values.argmax() != (q_values+ucb).argmax():
                ucb_work += 1
        else:
            ucb = None

        action = epsilon_greedy_act(num_actions, state, model,
                                    eps_t, ucb=ucb)
        episode_steps = 0
        max_episode_state = 0
        while True:
            next_state, rew, done, _ = env.step(action)
            if add_bonus:
                if torch.cuda.is_available():
                    e_values = e_model.forward(convert_to_var(state)).cpu().data.numpy()
                else:
                    e_values = e_model.forward(convert_to_var(state)).data.numpy()
                cnt = np.log(e_values) / np.log(1 - e_lr) + np.log(2) / np.log(1 - e_lr)
                rew_ = rew + beta / cnt[action]
            else:
                rew_ = rew

            sum_rewards_per_episode[-1] += rew
            list_rewards_per_episode[-1].append(rew)
            state_history.append(state)
            t += 1

            if add_ucb and t>learning_starts_in_steps:
                if torch.cuda.is_available():
                    e_values = e_model.forward(convert_to_var(state)).cpu().data.numpy()
                    q_values = model.forward(convert_to_var(state)).cpu().data.numpy()
                else:
                    e_values = e_model.forward(convert_to_var(state)).data.numpy()
                    q_values = model.forward(convert_to_var(state)).data.numpy()
                cnt = np.log(e_values) / np.log(1 - e_lr) + np.log(2) / np.log(1-e_lr)
                ucb = np.sqrt(2 * np.log(t) / cnt)
                ucb *= beta
                if q_values.argmax() != (q_values+ucb).argmax():
                    t_ucb_max = t
                    ucb_work += 1
                else:
                    pass
            else:
                ucb = None
            if act_type == 'epsilon_greedy':
                next_action = epsilon_greedy_act(num_actions, state, model, eps_t, ucb=ucb)
            elif act_type == 'lll_epsilon_greedy':
                next_action = lll_epsilon_greedy_act(num_actions, state, model, e_model, e_lr,
                                                     eps_t, ucb=ucb)

            replay_buffer.add(state, action, rew_, done, next_state, next_action)
            trajectory.append((state, action, rew_, done, next_state, next_action))

            batch = [np.array([state]), np.array([action]), np.array([rew_]),
                     np.array([done]), np.array([next_state]), np.array([next_action])]
            e_loss = sarsa_loss(e_optimizer, e_model, target_e_model, batch, gamma_E)

            state = next_state
            action = next_action

            if t > learning_starts_in_steps and t % train_freq_in_steps == 0:
                batch = replay_buffer.sample(batch_size)
                loss = dqn_loss(optimizer, model, target_model, batch, gamma,
                                target_type=target_type)
            else:
                loss = 0

            if t > learning_starts_in_steps and t % e_train_freq_in_steps == 0:
                batch = replay_buffer.sample(batch_size)
                e_loss = sarsa_loss(e_optimizer, e_model, target_e_model, batch, gamma_E)
            else:
                e_loss = 0

            if t > learning_starts_in_steps and t % update_freq_in_steps == 0:
                target_model = copy.deepcopy(model)

            if t > learning_starts_in_steps and t % e_update_freq_in_steps == 0:
                target_e_model = copy.deepcopy(e_model)

            if write_logs:
                write_tensorboard_logs(locals())
            episode_steps += 1
            if done:
                if print_freq is not None:
                    print('t: {}, Episode: {}, sum: {:.2f}, eps: {:.2f}'.\
                          format(t, num_episodes, sum_rewards_per_episode[-1], eps_t))
                    if add_ucb:
                        print('ucb in episode: {}'.format(ucb_work))

                num_episodes += 1
                sum_rewards_per_episode.append(0)
                state_history.append(next_state)
                state = env.reset()
                break

    state_history = np.array(state_history)
    if return_trajectory:
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        next_actions = []
        for t in trajectory:
            state, action, rew_, done, next_state, next_action = t
            states.append(state)
            actions.append(action)
            rewards.append(rew_)
            dones.append(done)
            next_states.append(next_state)
            next_actions.append(next_action)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        next_states = np.array(next_states)
        next_actions = np.array(next_actions)

        np.save(trajectory_path+'/_states', states)
        np.save(trajectory_path + '/_actions', actions)
        np.save(trajectory_path + '/_rewards', rewards)
        np.save(trajectory_path + '/_dones', dones)
        np.save(trajectory_path + '/_next_states', next_states)
        np.save(trajectory_path + '/_next_actions', next_actions)

    if return_states:
        if done:
            return sum_rewards_per_episode[:-1], num_episodes-1, state_history
        else:
            return sum_rewards_per_episode, num_episodes, state_history
    else:
        if done:
            return sum_rewards_per_episode[:-1], num_episodes-1
        else:
            return sum_rewards_per_episode, num_episodes
