############################# ViT_modules ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: creating a ViT model

########################## API Section #########################
import torch
from torch import nn
from SOTAs_main_func.SIL_Model.ViT_models.ViT import VisionTransformer

########################## API Section #########################
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

########################## API Section #########################
class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

########################## API Section #########################
class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

########################## API Section #########################
class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

########################## API Section #########################
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

########################## API Section #########################
def creating_ViT(class_num = 3, gpu_device = 0):
    vit_base = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=class_num, embed_dim=768,
                                depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
                                         norm_layer=nn.LayerNorm)

    vit_weight_file = torch.load(r'E:\SOTA_Model_Classification\SC_Weights\ViT\jx_vit_base_p16_224-80ecf9dd.pth')
    vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'head' not in k)}
    # vit_weight_file = {k: v for k, v in vit_weight_file.items() if
    #                   (k in vit_weight_file and 'patch_embed.proj' not in k)}
    vit_base.load_state_dict(vit_weight_file, strict=False)
    vit_net = ViT_Net(base_net=vit_base, class_num=class_num)
    nn.init.trunc_normal_(vit_net.head.weight)
    vit_net.cuda(gpu_device)
    return vit_net

########################## API Section #########################
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

########################## API Section #########################
class ViT_Net(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(ViT_Net, self).__init__()
        self.patch_embed = base_net.patch_embed
        self.cls_token = base_net.cls_token
        self.pos_embed = base_net.pos_embed
        self.pos_drop = base_net.pos_drop
        self.transformer_block_0 = base_net.blocks[0]
        self.transformer_block_1 = base_net.blocks[1]
        self.transformer_block_2 = base_net.blocks[2]
        self.transformer_block_3 = base_net.blocks[3]
        self.transformer_block_4 = base_net.blocks[4]
        self.transformer_block_5 = base_net.blocks[5]
        self.transformer_block_6 = base_net.blocks[6]
        self.transformer_block_7 = base_net.blocks[7]
        self.transformer_block_8 = base_net.blocks[8]
        self.transformer_block_9 = base_net.blocks[9]
        self.transformer_block_10 = base_net.blocks[10]
        self.transformer_block_11 = base_net.blocks[11]
        self.norm = base_net.norm
        self.head = base_net.head
        self.evd_linear = nn.Linear(in_features=768, out_features=768)

    def forward(self, x):
        y = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        y  = torch.cat((cls_tokens, y), dim=1)
        y = y + self.pos_embed
        y = self.pos_drop(y)
        y = self.transformer_block_0(y)
        y = self.transformer_block_1(y)
        y = self.transformer_block_2(y)
        y = self.transformer_block_3(y)
        y = self.transformer_block_4(y)
        y = self.transformer_block_5(y)
        y = self.transformer_block_6(y)
        y = self.transformer_block_7(y)
        y = self.transformer_block_8(y)
        y = self.transformer_block_9(y)
        y = self.transformer_block_10(y)
        y = self.transformer_block_11(y)
        #y_sym = y @ y.permute(0, 2, 1)
        #_, u = torch.linalg.eigh(y_sym)
        #y = y + (u.permute(0, 2, 1) @ y)
        y = self.norm(y)
        y = self.head(y)
        return y[:, 0, :]

########################## API Section #########################
class ViT_Small_Net(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(ViT_Small_Net, self).__init__()
        self.patch_embed = base_net.patch_embed
        self.cls_token = base_net.cls_token
        self.pos_embed = base_net.pos_embed
        self.pos_drop = base_net.pos_drop
        self.transformer_block_0 = base_net.blocks[0]
        self.transformer_block_1 = base_net.blocks[1]
        self.transformer_block_2 = base_net.blocks[2]
        self.transformer_block_3 = base_net.blocks[3]
        self.transformer_block_4 = base_net.blocks[4]
        self.transformer_block_5 = base_net.blocks[5]
        self.transformer_block_6 = base_net.blocks[6]
        self.transformer_block_7 = base_net.blocks[7]
        self.norm = base_net.norm
        self.head = base_net.head

    def forward(self, x):
        y = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        y  = torch.cat((cls_tokens, y), dim=1)
        y = y + self.pos_embed
        y = self.pos_drop(y)
        y = self.transformer_block_0(y)
        y = self.transformer_block_1(y)
        y = self.transformer_block_2(y)
        y = self.transformer_block_3(y)
        y = self.transformer_block_4(y)
        y = self.transformer_block_5(y)
        y = self.transformer_block_6(y)
        y = self.transformer_block_7(y)
        y = self.norm(y)
        y = self.head(y)
        return y[:, 0, :]

########################## API Section #########################
class ViT_Large_Net(nn.Module):
    def __init__(self, class_num=None, base_net=None):
        super(ViT_Large_Net, self).__init__()
        self.patch_embed = base_net.patch_embed
        self.cls_token = base_net.cls_token
        self.pos_embed = base_net.pos_embed
        self.pos_drop = base_net.pos_drop
        self.transformer_block_0 = base_net.blocks[0]
        self.transformer_block_1 = base_net.blocks[1]
        self.transformer_block_2 = base_net.blocks[2]
        self.transformer_block_3 = base_net.blocks[3]
        self.transformer_block_4 = base_net.blocks[4]
        self.transformer_block_5 = base_net.blocks[5]
        self.transformer_block_6 = base_net.blocks[6]
        self.transformer_block_7 = base_net.blocks[7]
        self.transformer_block_8 = base_net.blocks[8]
        self.transformer_block_9 = base_net.blocks[9]
        self.transformer_block_10 = base_net.blocks[10]
        self.transformer_block_11 = base_net.blocks[11]
        self.transformer_block_12 = base_net.blocks[12]
        self.transformer_block_13 = base_net.blocks[13]
        self.transformer_block_14 = base_net.blocks[14]
        self.transformer_block_15 = base_net.blocks[15]
        self.transformer_block_16 = base_net.blocks[16]
        self.transformer_block_17 = base_net.blocks[17]
        self.transformer_block_18 = base_net.blocks[18]
        self.transformer_block_19 = base_net.blocks[19]
        self.transformer_block_20 = base_net.blocks[20]
        self.transformer_block_21 = base_net.blocks[21]
        self.transformer_block_22 = base_net.blocks[22]
        self.transformer_block_23 = base_net.blocks[23]
        self.norm = base_net.norm
        self.head = base_net.head

    def forward(self, x):
        y = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        y  = torch.cat((cls_tokens, y), dim=1)
        y = y + self.pos_embed
        y = self.pos_drop(y)
        y = self.transformer_block_0(y)
        y = self.transformer_block_1(y)
        y = self.transformer_block_2(y)
        y = self.transformer_block_3(y)
        y = self.transformer_block_4(y)
        y = self.transformer_block_5(y)
        y = self.transformer_block_6(y)
        y = self.transformer_block_7(y)
        y = self.transformer_block_8(y)
        y = self.transformer_block_9(y)
        y = self.transformer_block_10(y)
        y = self.transformer_block_11(y)
        y = self.transformer_block_12(y)
        y = self.transformer_block_13(y)
        y = self.transformer_block_14(y)
        y = self.transformer_block_15(y)
        y = self.transformer_block_16(y)
        y = self.transformer_block_17(y)
        y = self.transformer_block_18(y)
        y = self.transformer_block_19(y)
        y = self.transformer_block_20(y)
        y = self.transformer_block_21(y)
        y = self.transformer_block_22(y)
        y = self.transformer_block_23(y)
        y = self.norm(y)
        y = self.head(y)
        return y[:, 0, :]


########################## API Section #########################
class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        y = self.softmax(x)
        return y


