############################# SwinT_model_modules ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: creating SwinT model by loading pretrained weight for searching the best learning rate

########################## API Section #########################
from SOTAs_main_func.SIL_Model.SwinT_models.models.swin_transformer import SwinTransformer
from torch import nn
import torch

class SwinT_Net(nn.Module):
    def __init__(self, base_model=None, class_num = 3):
        super(SwinT_Net, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
        self.head = base_model.head

    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)
        y = self.norm(y)
        y = y.permute(0, 2, 1)
        y = self.avgp(y)
        y = self.head(y.reshape(y.shape[0], y.shape[1]))
        return y

class IDANet(nn.Module):
    def __init__(self, base_model=None, class_num = 3):
        super(IDANet, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
        self.head = base_model.head
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.head_2 = nn.Linear(in_features=768, out_features=class_num)
        self.norm_2 = base_model.norm

    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)

        u, lamda, v = torch.linalg.svd(y)
        u_diags = torch.zeros((v.shape[0], lamda.shape[1], v.shape[2]), device=u.device)
        for i in range(u_diags.shape[0]):
            for j in range(lamda.shape[1]):
                u_diags[i, j, j] = lamda[i, j]

        y_const = u[:, :, 0:1] @ u_diags[:, 0:1, :] @ v
        y_const = self.norm_2(y_const)
        y_const = self.avgp(y_const.permute(0, 2, 1))
        y_const = self.head_2(y_const.reshape(y_const.shape[0], y_const.shape[1]))

        y = self.norm(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = self.head(y.reshape(y.shape[0], y.shape[1]))
        y_3 = self.softmax(y) + self.softmax(y_const)
        return y, y_const, y_3

class IDANet_Visulization(nn.Module):
    def __init__(self, base_model=None, class_num = 3):
        super(IDANet_Visulization, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
        self.head = base_model.head
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.head_2 = nn.Linear(in_features=768, out_features=class_num)
        self.norm_2 = nn.LayerNorm(normalized_shape=(768))

    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)

        u, lamda, v = torch.linalg.svd(y)
        u_diags = torch.zeros((v.shape[0], lamda.shape[1], v.shape[2]), device=u.device)
        for i in range(u_diags.shape[0]):
            for j in range(lamda.shape[1]):
                u_diags[i, j, j] = lamda[i, j]

        y_const = u[:, :, 0:1] @ u_diags[:, 0:1, :] @ v
        y_const = self.norm_2(y_const)
        y_const = self.avgp(y_const.permute(0, 2, 1))
        y_const = self.head_2(y_const.reshape(y_const.shape[0], y_const.shape[1]))

        y = self.norm(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = self.head(y.reshape(y.shape[0], y.shape[1]))
        y_3 = self.softmax(y) + self.softmax(y_const)
        return y, y_const, y_3

class IDANet_Ablation(nn.Module):
    def __init__(self, base_model=None, class_num = 3):
        super(IDANet_Ablation, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
        self.head = base_model.head
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.head_2 = nn.Linear(in_features=768, out_features=class_num)
        self.head_3 = nn.Linear(in_features=768, out_features=class_num)

    def forward(self, x):
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)

        u, lamda, v = torch.linalg.svd(y)
        u_diags = torch.zeros((v.shape[0], lamda.shape[1], v.shape[2]), device=u.device)
        for i in range(u_diags.shape[0]):
            for j in range(lamda.shape[1]):
                u_diags[i, j, j] = lamda[i, j]

        y_r = u[:, :, 0:1] @ u_diags[:, 0:1, :] @ v
        y_r = self.norm(y_r)
        y_r = self.avgp(y_r.permute(0, 2, 1))
        y_r = self.head_2(y_r.reshape(y_r.shape[0], y_r.shape[1]))

        u_diags_t = u_diags.permute(0, 2, 1)
        y_c = v[:, :, 0:1] @ u_diags_t[:, 0:1, :] @ u
        y_c = y_c.permute(0, 2, 1)
        y_c = self.norm(y_c)
        y_c = self.avgp(y_c.permute(0, 2, 1))
        y_c = self.head_3(y_c.reshape(y_c.shape[0], y_c.shape[1]))

        y = self.norm(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = self.head(y.reshape(y.shape[0], y.shape[1]))
        y_3 = self.softmax(y) + self.softmax(y_r)
        return y, y_r, y_3

def creating_swinT(gpu_device=0, class_num=3):
    swinT_base = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=class_num,
                                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                 use_checkpoint=False, fused_window_process=False)

    checkpoint = torch.load(r'E:\SOTA_Model_Classification\SC_Weights\SwinT\swin_tiny_patch4_window7_224_22k.pth',
                            map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = swinT_base.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            pass#logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = swinT_base.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            pass
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = swinT_base.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            #logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(swinT_base.head.bias, 0.)
            torch.nn.init.constant_(swinT_base.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']

    swinT_base.load_state_dict(state_dict, strict=False)

    nn.init.trunc_normal_(swinT_base.head.weight, std=.02)
    print(swinT_base.layers[0].blocks[0].mlp.fc2.weight)

    #with torch.no_grad():
    #    print('########################## SwinT_summary #########################')
    #    summary(swinT_base, torch.randn((1, 3, 224, 224)))
    #    print('\n', '########################## SwinT_net #########################')
    #    print(swinT_base, '\n')

    swinT_base = swinT_base.cuda(gpu_device)
    return swinT_base