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
        ns = 3 * len + 2
        TabularEnvBase.__init__(self, ns, num_actions, one_hot)

        self.initial_rew = 1
        self.side_rew = -100
        self.target_rew = 10

    # rewards
    # =====10===== #
    # -100 0 -100  #
    # ...........  #
    # -100 0 -100  #
    # =====1====== #

    # state ids
    # = 7 = #
    # 4 5 6 #
    # 1 2 3 #
    # = 0 = #

    def step(self, a):
        # 0 down
        # 1 left
        # 2 right
        # 3 up
        self.count_steps += 1
        reward = 0

        if a == 0:
            if self.cur_state_id == 0:
                reward = self.initial_rew
            elif self.cur_state_id == 1 or self.cur_state_id == 3:
                reward = self.side_rew
            elif self.cur_state_id == 2:
                reward = 1
                self.cur_state_id = 0
            elif self.cur_state_id % 3 == 1 or self.cur_state_id % 3 == 0:
                reward = self.side_rew
                self.cur_state_id -= 3
            elif self.cur_state_id % 3 == 2:
                reward = 0
                self.cur_state_id -= 3

        if a == 1:
            if self.cur_state_id == 0:
                reward = self.initial_rew
            elif self.cur_state_id == self.ns-1:
                reward = self.target_rew
            elif self.cur_state_id % 3 == 0:
                reward = 0
                self.cur_state_id -= 1
            elif self.cur_state_id % 3 == 1:
                reward = self.side_rew
            elif self.cur_state_id % 3 == 2:
                reward = self.side_rew
                self.cur_state_id -= 1

        if a == 2:
            if self.cur_state_id == 0:
                reward = self.initial_rew
            elif self.cur_state_id == self.ns-1:
                reward = self.target_rew
            elif self.cur_state_id % 3 == 0:
                reward = self.side_rew
            elif self.cur_state_id % 3 == 1:
                reward = 0
                self.cur_state_id += 1
            elif self.cur_state_id % 3 == 2:
                reward = self.side_rew
                self.cur_state_id += 1

        if a == 3:
            if self.cur_state_id == 0:
                reward = 0
                self.cur_state_id = 2
            elif self.cur_state_id == self.ns -3 or self.cur_state_id == self.ns-1:
                reward = self.target_rew
                self.cur_state_id = self.ns-1
            elif self.cur_state_id == self.ns-4 or self.cur_state_id == self.ns-2:
                reward = self.side_rew
            elif self.cur_state_id % 3 == 1 or self.cur_state_id % 3 == 0:
                reward = self.side_rew
                self.cur_state_id += 3
            elif self.cur_state_id % 3 == 2:
                reward = 0
                self.cur_state_id += 3

        self.reward += reward
        self.cur_state_descr = self.convert_ns_to_description(state_id=self.cur_state_id)

        #if self.count_steps == self.len + 10:
        if self.cur_state_id == self.ns - 1:
            done = True
        else:
            done = False

        s, r, d, _ = self.cur_state_descr, reward, done, None

        if done:
            self.reset()


        return s, r, d, _

