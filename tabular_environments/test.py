from tabular_environments.flipping_chain_environment import FlippingChain

env = FlippingChain()
s = env.reset()

cnt = 0
total_rew = 0
for t in range(2000 * 1000):
    s, r, done, _ = env.step(1)
    cnt+=1
    total_rew+=r
print('--')
print(cnt)
print(total_rew/2000)