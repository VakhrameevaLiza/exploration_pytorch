from tabular_environments.base_tabular_env import TabularEnvBase
import numpy as np


class ActionSpace:
    def __init__(self, na):
        self.n = na


class ObeservationSpace:
    def __init__(self, ns):
        self.shape = (ns,)


class FlippingChain(TabularEnvBase):
    def __init__(self, ns=5, num_actions=2, one_hot=False):
        TabularEnvBase.__init__(self, ns, num_actions, one_hot)
        self.max_rew = 10
        self.min_rew = self.max_rew / ns

    def step(self, a):
        # 1 forward or loop
        # 0 backward or loop
        self.count_steps += 1
        reward = 0

        if np.random.rand() < 0.2:
            if a == 1:
                a = 0
            else:
                a = 1

        if a == 1:
            if self.cur_state_id == self.ns - 1:
                reward = self.max_rew
            self.cur_state_id = min(self.cur_state_id + 1, self.ns - 1)
        else:
            reward = self.min_rew
            self.cur_state_id = 0

        self.reward += reward
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)

        if self.count_steps == self.ns + 10:
            done = True
        else:
            done = False

        s, r, d, _ = self.cur_state_descr, reward, done, None

        if done:
            self.reset()

        return s, r, d, _

