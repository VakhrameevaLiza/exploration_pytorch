from tabular_environments.flipping_chain_environment import FlippingChain

env = FlippingChain()
s = env.reset()
print(s)
actions = [1,1,1,1,1]
for a in actions:
    s, _, _, _ = env.step(a)
    print(s)