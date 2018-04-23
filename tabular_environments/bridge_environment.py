from tabular_environments.base_tabular_env import TabularEnvBase
import numpy as np


class ActionSpace:
    def __init__(self, na):
        self.n = na


class ObeservationSpace:
    def __init__(self, ns):
        self.shape = (ns,)


class Bridge(TabularEnvBase):
    def __init__(self, len, num_actions=4, one_hot=False):
        self.len = len
        ns = 3 * len
        TabularEnvBase.__init__(self, ns, num_actions, one_hot)

        self.initial_rew = 1
        self.side_rew = -100
        self.target_rew = 10

        self.cur_state_id = 1
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)


    def reset(self):
        self.cur_state_id = 1
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)
        self.count_steps = 0
        self.reward = 0
        return self.cur_state_descr


    # rewards
    # -100 10 -100  #
    # -100 0  -100  #
    # ............  #
    # -100 1  -100  #

    # state ids

    # 6 7 8
    # 3 4 5
    # 0 1 2

    def step(self, a):
        # 0 down
        # 1 left
        # 2 right
        # 3 up
        self.count_steps += 1
        reward = 0
        done = False

        if a == 0:
            if self.cur_state_id == 1:
                reward = self.initial_rew
            else:
                self.cur_state_id -= 3
                reward = 0

        if a == 1:
            self.cur_state_id -= 1
            reward = self.side_rew
            done = True

        if a == 2:
            self.cur_state_id += 1
            reward = self.side_rew
            done = True

        if a == 3:
            if self.cur_state_id == self.ns - 2:
                reward = self.target_rew
            else:
                self.cur_state_id += 3
                reward = 0

        self.reward += reward
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)

        if self.count_steps == self.len + 9 or done:
            done = True

        s, r, d, _ = self.cur_state_descr, reward, done, None

        if done:
            self.reset()

        return s, r, d, _

