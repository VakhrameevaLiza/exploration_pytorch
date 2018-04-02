from tabular_environments.base_tabular_env import TabularEnvBase
import numpy as np


class ActionSpace:
    def __init__(self, na):
        self.n = na


class ObeservationSpace:
    def __init__(self, ns):
        self.shape = (ns,)


class SimpleChain(TabularEnvBase):
    def __init__(self, ns, num_actions=2, one_hot=False):
        TabularEnvBase.__init__(self, ns, num_actions, one_hot)
        self.eps = 1e-3

    def step(self, a):
        # 1 forward or loop
        # 0 backward or loop
        self.count_steps += 1
        reward = 0

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

        if done:
            self.reset()

        return s, r, d, _

