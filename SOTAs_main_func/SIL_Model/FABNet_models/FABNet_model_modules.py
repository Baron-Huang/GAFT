############################# FABNet.py ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: the modules of FABNet

########################## API Section #########################
from torch import nn
import torch

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Dense_Net(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(Dense_Net, self).__init__()
        self.base_net = base_net.features
        self.top_AvgMp = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
        self.top_flat = nn.Flatten()
        self.top_linear_3 = nn.Linear(1024, class_num)
        #nn.init.kaiming_normal(self.top_linear_3.weight)

    def forward(self, x):
        y = self.base_net(x)
        y = self.top_AvgMp(y)
        y = self.top_flat(y)
        y = self.top_linear_3(y)
        return y

class FABNet(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(FABNet, self).__init__()
        #self.base_net = base_net.features
        self.input_module = base_net.features[0:4]
        self.dense_block_1 = base_net.features.denseblock1
        self.dense_block_2 = base_net.features.denseblock2
        self.dense_block_3 = base_net.features.denseblock3
        self.dense_block_4 = base_net.features.denseblock4
        self.transition_1 = base_net.features.transition1
        self.transition_2 = base_net.features.transition2
        self.transition_3 = base_net.features.transition3
        self.BN_5 = base_net.features.norm5

        self.top_AvgMp = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
        self.top_flat = nn.Flatten()
        #self.top_linear_1 = nn.Linear(1024 * 7 * 7, 1024)
        self.top_linear_3 = nn.Linear(1024, class_num)
        self.dp = nn.Dropout(p=0.3)
        self.SE_1 = SELayer(128, 8)
        self.SE_2 = SELayer(256, 16)
        self.SE_3 = SELayer(512, 24)
        self.SE_4 = SELayer(1024, 32)
        self.NonLocal_1 = NONLocalBlock2D(in_channels=128)
        self.NonLocal_2 = NONLocalBlock2D(in_channels=256)
        self.NonLocal_3 = NONLocalBlock2D(in_channels=512)
        self.NonLocal_4 = NONLocalBlock2D(in_channels=1024)
        #nn.init.kaiming_normal(self.top_linear_3.weight)

    def forward(self, x):
        #y = self.base_net(x)
        y = self.input_module(x)
        y = self.dense_block_1(y)
        y = self.transition_1(y)
        y_1 = self.SE_1(y)
        y_2 = self.NonLocal_1(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_2(y)
        y = self.transition_2(y)
        y_1 = self.SE_2(y)
        y_2 = self.NonLocal_2(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_3(y)
        y = self.transition_3(y)
        y_1 = self.SE_3(y)
        y_2 = self.NonLocal_3(y)
        y = (y_1 + y_2) / 2.0
        y = self.dense_block_4(y)
        y = self.BN_5(y)
        y_1 = self.SE_4(y)
        y_2 = self.NonLocal_4(y)
        y = (y_1 + y_2) / 2.0
        y = self.top_AvgMp(y)
        y = self.top_flat(y)
        #y = self.dp(y)
        #y = self.top_linear_1(y)
        #y = self.top_linear_2(y)
        y = self.top_linear_3(y)
        return y
