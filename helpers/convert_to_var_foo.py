import torch
from torch.autograd import Variable
import numpy as np


def convert_to_var(arr, astype='float32', add_dim=False):
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    if add_dim:
        arr = np.array([arr])
    if astype == 'float32':
        v = Variable(FloatTensor(arr.astype(astype)))
    else:
        v = Variable(LongTensor(arr.astype(astype)))
    return v