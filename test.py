import gym

def foo(env):
    env.reset()
    for _ in range(10000):
        env.render()
        env.step(env.action_space.sample())  # take a random action

env = gym.make('MountainCar-v0')
foo(env)
