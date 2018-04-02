import numpy as np


class ActionSpace:
    def __init__(self, na):
        self.n = na


class ObeservationSpace:
    def __init__(self, ns):
        self.shape = (ns,)


class SimpleChain:
    def __init__(self, ns, p=0.0, eps=0.01, one_hot=False):
        self.ns = ns
        self.action_space = ActionSpace(2)
        self.observation_space = ObeservationSpace(self.ns)
        self.eps = eps
        self.one_hot=one_hot
        self.p = p

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

    def step(self, a):
        # 0 forward or loop
        # 1 backward or loop
        self.count_steps += 1
        reward = 0

        if a == 1 and np.random.rand() < self.p:
            a = 0

        if a == 1:
            if self.cur_state_id == self.ns - 1:
                reward = 1
            self.cur_state_id = min(self.cur_state_id + 1, self.ns - 1)
        else:
            if self.cur_state_id == 0:
                reward = self.eps
            self.cur_state_id = max(self.cur_state_id - 1, 0)

        self.reward += reward
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)

        if self.count_steps == self.ns + 9:
            done = True
        else:
            done = False

        s, r, d, _ = self.cur_state_descr, reward, done, None

        #if done:
        #    self.reset()

        return s, r, d, _

