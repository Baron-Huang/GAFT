import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

class CORAL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source, target, **kwargs):
        #source = torch.squeeze(source)  #+
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]

        source = source.to(source.device)  # 将 source 张量移动到与 target 张量相同的设备上
        target = target.to(source.device)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        target = torch.squeeze(target)  #+
        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4*d*d)
        return loss