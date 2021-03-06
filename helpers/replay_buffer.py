import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size, seed=None):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        if seed:
            random.seed(seed)

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, done, obs_tp1, action_tp1):
        data = (obs_t, action, reward, done, obs_tp1, action_tp1)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._next_idx = int(self._next_idx)

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, dones, obses_tp1, actions_tp1 = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, done, obs_tp1, action_tp1 = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            dones.append(done)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            actions_tp1.append(np.array(action_tp1, copy=False))
        return np.array(obses_t), np.array(actions), np.array(rewards), \
               np.array(dones), np.array(obses_tp1), np.array(actions_tp1)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class CountBasedReplayBuffer(object):
    def __init__(self, size, reward_type, state_to_id, seed=None):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.reward_type = reward_type
        self.state_to_id = state_to_id
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        if seed:
            random.seed(seed)

    def _counter_to_reward(self, counter, obs, action):
        eps = np.finfo(float).eps
        obs_id = self.state_to_id(obs)
        if self.reward_type == 'state':
            rew = 1 / (np.sqrt(counter[obs_id].sum()) + 1)
        else:
            rew = 1 / (np.sqrt(counter[obs_id][action]) + 1)
        return rew

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, obs_tp1, done):
        data = (obs_t, action, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes, s_a_visitations_count):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            reward = self._counter_to_reward(s_a_visitations_count, obs_t, action)
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, s_a_visitations_count):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, s_a_visitations_count)