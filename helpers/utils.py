import numpy as np
import random
import torch

def set_seed(seed, env=None):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.seed(seed)