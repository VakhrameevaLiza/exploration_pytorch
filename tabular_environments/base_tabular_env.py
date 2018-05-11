import numpy as np


class ActionSpace:
    def __init__(self, na):
        self.n = na


class ObeservationSpace:
    def __init__(self, ns):
        self.shape = (ns,)


class TabularEnvBase:
    def __init__(self, ns, num_actions, one_hot=False):
        self.ns = ns
        self.action_space = ActionSpace(num_actions)
        self.observation_space = ObeservationSpace(self.ns)
        self.one_hot=one_hot

        all_states = np.ones((self.ns, self.ns)) * (np.arange(self.ns)[:, np.newaxis] + 1) / 100
        self.all_states = (all_states - all_states.mean())

        self.cur_state_id = 0
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)
        self.count_steps = 0
        self.reward = 0

    def get_all_states(self):
        return self.all_states

    def convert_ns_to_description(self, state_id):
        return self.all_states[state_id]

    def reset(self):
        self.cur_state_id = 0
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)
        self.count_steps = 0
        self.reward = 0
        return self.cur_state_descr

    def convert_state_to_id(self, s):
        for i, cur_s in enumerate(self.all_states):
            if np.array_equal(cur_s, s):
                return i

    def convert_state_to_id_func(self):
        if self.one_hot:
            return lambda x: int(x.argmax())
        else:
            return lambda x: int(x.sum()-1)

    def seed(self, seed):
        pass