import numpy as np
import torch
from torch import nn
from .model import WikiModel
import os


def generation(inp, forward):
    inp_tensor = torch.from_numpy(inp.T).float().type('torch.LongTensor')
    inp_var = torch.autograd.Variable(inp_tensor)
    charcount = 33278
    EMBEDDING_DIM = 400
    HIDDEN_SIZE = 1150
    savepath = os.path.abspath(os.path.join(__file__, '../params.pt'))
    pre_net = WikiModel(charcount, EMBEDDING_DIM, HIDDEN_SIZE)
    pre_net.load_state_dict(
        torch.load(savepath, map_location=lambda storage, loc: storage))
    pre_net.eval()
    classes = pre_net(inp_var, forward=forward).numpy().T
    return classes
