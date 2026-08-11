import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

class BNM(nn.Module):
    """ Batch nuclear-norm maximization, CVPR 2020.
    tar: a tensor, softmax target output.
    NOTE: this does not require source domain data.
    """
    def __init__(self):
        super().__init__()

    def forward(self, target):
        _, out, _ = torch.svd(target)
        loss = -torch.mean(out)#torch.abs(-torch.mean(out))
        return loss
