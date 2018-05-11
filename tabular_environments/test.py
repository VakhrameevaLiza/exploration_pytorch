from tabular_environments.flipping_chain_environment import FlippingChain
from tabular_environments.chain_environment import SimpleChain
import numpy as np

env = SimpleChain(5)

print(env.get_all_states())