from tabular_environments.chain_environment import SimpleChain
from tabular_environments.bridge_environment import Bridge

env = Bridge(2)
s = env.reset()
actions = [3, 3, 1, 1, 0]
for a in actions:
    t = env.step(a)
    print(t[0])

# 0 down
# 1 left
# 2 right
# 3 up


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
