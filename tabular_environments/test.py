from tabular_environments.flipping_chain_environment import FlippingChain
from tabular_environments.bridge_environment import Bridge
from tabular_environments.chain_environment import SimpleChain

env = Bridge(10)
s = env.reset()
cnt = 0
total_rew = 0
"""
steps = [3, 3, 2]

for a in steps:
    s, r, done, _ = env.step(a)
    print(env.convert_state_to_id(s), r, done)
"""
while True:
    s, r, done, _ = env.step(3)
    print(env.convert_state_to_id(s))
    cnt+=1
    total_rew+=r
    if done:
        break
print('--')
print(cnt)
print(total_rew)

