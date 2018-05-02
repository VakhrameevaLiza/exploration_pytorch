import gym


class SparseAcrobot:
    def __init__(self):
        self.env = gym.make('Acrobot-v1')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.name = 'acrobot'
        self.steps_per_episode = 0

    def step(self, action):
        self.steps_per_episode += 1
        next_state, reward, done, info = self.env.step(action)
        if done and self.steps_per_episode < 500:
            reward = 1
        else:
            reward = 0
        return next_state, reward, done, info

    def reset(self):
        self.steps_per_episode = 0
        return self.env.reset()

    def seed(self, seed):
        self.env.seed(seed)


"""
class SparseCartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.name = 'cartpole'
        self.steps_per_episode = 0

    def step(self, action):
        self.steps_per_episode += 1
        next_state, reward, done, info = self.env.step(action)
        if done and self.steps_per_episode < 500:
            reward = 1
        else:
            reward = 0
        return next_state, reward, done, info

    def reset(self):
        self.steps_per_episode = 0
        return self.env.reset()

    def seed(self, seed):
        self.env.seed(seed)

"""

class SparseMountainCar:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.name = 'mountain_car'
        self.steps_per_episode = 0

    def step(self, action):
        self.steps_per_episode += 1
        next_state, reward, done, info = self.env.step(action)
        if done and self.steps_per_episode < 200:
            reward = 1
        else:
            reward = 0
        return next_state, reward, done, info

    def reset(self):
        self.steps_per_episode = 0
        return self.env.reset()

    def seed(self, seed):
        self.env.seed(seed)