#-*- utf-8 -*-

'''本程序用于验证hook编程获取卷积层的输出特征图和特征图的梯度'''

__author__ = 'puxitong from UESTC'

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import warnings
from torchsummaryX import summary
warnings.filterwarnings('ignore')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,3,1,1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,9,3,1,1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(8*8*9, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120,10)

    def forward(self, x):
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.view(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)

        return out

def backward_hook(module, grad_in, grad_out):
    grad_block['grad_in'] = grad_in
    grad_block['grad_out'] = grad_out

def farward_hook(module, inp, outp):
    fmap_block['input'] = inp
    fmap_block['output'] = outp

if __name__ == '__main__':
    loss_func = nn.CrossEntropyLoss()

    # 生成一个假标签以便演示
    label = torch.empty(1, dtype=torch.long).random_(3)

    # 生成一副假图像以便演示
    input_img = torch.randn(1,3,32,32).requires_grad_()


    fmap_block = dict()  # 装feature map
    grad_block = dict()  # 装梯度

    net = Net()
    print(summary(net, input_img))
    print(net)
    # 注册hook
    net.conv2.register_forward_hook(farward_hook)
    net.conv2.register_backward_hook(backward_hook)

    outs = net(input_img)
    loss = loss_func(outs, label)
    loss.backward()


    print(fmap_block['output'][0].shape)
    print(grad_block['grad_out'])