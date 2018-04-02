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

        self.cur_state_id = 0
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)
        self.count_steps = 0
        self.reward = 0

    def get_all_states(self):
        if self.one_hot:
            all_states = np.eye(self.ns)
        else:
            all_states = np.tril(np.ones((self.ns, self.ns)))
        return all_states

    def convert_ns_to_description(self, state_id):
        state_descr = np.zeros(self.ns)
        if self.one_hot:
            state_descr[state_id] = 1
        else:
            state_descr[:state_id+1] = 1
        return state_descr

    def reset(self):
        self.cur_state_id = 0
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)
        self.count_steps = 0
        self.reward = 0
        return self.cur_state_descr

    def convert_state_to_id(self, s):
        if self.one_hot:
            return int(s.argmax())
        else:
            return int(s.sum()-1)

    def convert_state_to_id_func(self):
        if self.one_hot:
            return lambda x: int(x.argmax())
        else:
            return lambda x: int(x.sum()-1)