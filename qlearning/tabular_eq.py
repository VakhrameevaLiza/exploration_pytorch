import numpy as np


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


def Q_E_learning(env,
                 alpha=0.2, gamma=0.99, gamma_E=0.9,
                 ucb_flag=False, rew_bonus_flag=False, w=1,
                 max_num_episodes=2000):
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    Q = np.ones((n_states, n_actions)) / 1000
    E = np.ones((n_states, n_actions)) * 0.999999
    cnt = np.zeros((n_states, n_actions))

    eps = 0.75
    eps_decay = 0.95

    total_episodes_rew = [0]

    t = 0

    for episode in range(max_num_episodes):
        s = env.reset()

        s_id = env.convert_state_to_id(s)
        if np.random.rand() < eps:
            a = np.random.choice(n_actions)
        else:
            a = Q[s_id].argmax()

        while True:
            next_s, r, done, _ = env.step(a)
            cnt[s_id][a] += 1

            if rew_bonus_flag:
                r_bonus = 1 / np.sqrt(np.log(E[s_id][a]) / np.log(1 - alpha))
                r_ = r + w * r_bonus
            else:
                r_ = r

            total_episodes_rew[-1] += r
            next_s_id = env.convert_state_to_id(next_s)

            if np.random.rand() < eps:
                next_a = np.random.choice(n_actions)
            else:
                if ucb_flag:
                    e_cnt = np.log(E[next_s_id]) / np.log(1 - alpha)
                    ucb = np.sqrt(np.log(t) / e_cnt)
                else:
                    ucb = 0
                next_a = (Q[next_s_id] + ucb).argmax()

            Q[s_id][a] = (1 - alpha) * Q[s_id][a] + alpha * (r_ + gamma * Q[next_s_id].max() * (1 - done))
            E[s_id][a] = (1 - alpha) * E[s_id][a] + alpha * gamma_E * E[next_s_id][next_a] * (1 - done)

            s_id = next_s_id
            a = next_a
            t += 1

            if len(total_episodes_rew) > 100 and np.sum(total_episodes_rew[-100:]) == 10 * 100:
                return total_episodes_rew

            if done:
                eps *= eps_decay
                env.reset()
                total_episodes_rew.append(0)
                break

    return total_episodes_rew