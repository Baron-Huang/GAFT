import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import math
import torch.nn.functional as F
from torchinfo import summary
from SIL_Utils.fit_functions import Single_out_fit, testing_funnction



class DropPath(nn.Module):
    """Stochastic depth layer used by the official ViG implementation."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    if act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    if act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    if act == 'gelu':
        return nn.GELU()
    if act == 'hswish':
        return nn.Hardswish(inplace)
    raise NotImplementedError(f'Unsupported activation: {act}')


def norm_layer(norm, channels):
    if norm.lower() == 'batch':
        return nn.BatchNorm2d(channels, affine=True)
    if norm.lower() == 'instance':
        return nn.InstanceNorm2d(channels, affine=False)
    raise NotImplementedError(f'Unsupported normalization: {norm}')


class BasicConv(nn.Sequential):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        layers = []
        for index in range(1, len(channels)):
            layers.append(nn.Conv2d(channels[index - 1], channels[index], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                layers.append(norm_layer(norm, channels[index]))
            if act is not None and act.lower() != 'none':
                layers.append(act_layer(act))
        super().__init__(*layers)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def batched_index_select(x, index):
    batch_size, channels, reduced_vertices = x.shape[:3]
    _, vertices, neighbors = index.shape
    index_base = torch.arange(batch_size, device=index.device).view(-1, 1, 1) * reduced_vertices
    index = (index + index_base).reshape(-1)
    features = x.transpose(2, 1).contiguous().reshape(batch_size * reduced_vertices, -1)[index, :]
    return features.view(batch_size, vertices, neighbors, channels).permute(0, 3, 1, 2).contiguous()


def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(x * x, dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def dense_knn_matrix(x, k):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, points, _ = x.shape
        distance = pairwise_distance(x.detach())
        _, neighbors = torch.topk(-distance, k=k)
        centers = torch.arange(points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((neighbors, centers), dim=0)


class DenseDilated(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon

    def forward(self, edge_index):
        if self.stochastic and self.training and torch.rand(1, device=edge_index.device) < self.epsilon:
            indices = torch.randperm(self.k * self.dilation, device=edge_index.device)[:self.k]
            return edge_index[:, :, :, indices]
        return edge_index[:, :, :, ::self.dilation]


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        x = F.normalize(x, p=2.0, dim=1)
        return self.dilated(dense_knn_matrix(x, self.k * self.dilation))


class MRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super().__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j = torch.max(x_j - x_i, -1, keepdim=True)[0]
        batch_size, channels, vertices, _ = x.shape
        return self.nn(torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(batch_size, 2 * channels, vertices, 1))


class GraphConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, conv='mr', act='relu', norm=None, bias=True):
        super().__init__()
        if conv != 'mr':
            raise NotImplementedError(f'Unsupported graph convolution: {conv}')
        self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DyGraphConv2d(GraphConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='mr', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0):
        super().__init__(in_channels, out_channels, conv, act, norm, bias)
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        graph_input = x.reshape(batch_size, -1, height * width, 1).contiguous()
        edges = self.dilated_knn_graph(graph_input)
        output = super().forward(graph_input, edges)
        return output.reshape(batch_size, -1, height, width).contiguous()


class Grapher(nn.Module):
    def __init__(self, channels, kernel_size=9, dilation=1, act='relu', norm='batch', bias=True,
                 stochastic=False, epsilon=0.0, drop_path=0.0):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels))
        self.graph_conv = DyGraphConv2d(channels, channels * 2, kernel_size, dilation, 'mr', act, norm,
                                        bias, stochastic, epsilon)
        self.fc2 = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.BatchNorm2d(channels))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return self.drop_path(x) + shortcut


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features, act='relu', drop_path=0.0):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Conv2d(in_features, hidden_features, 1), nn.BatchNorm2d(hidden_features))
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_features, in_features, 1), nn.BatchNorm2d(in_features))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc2(self.act(self.fc1(x)))
        return self.drop_path(x) + shortcut


class Stem(nn.Module):
    def __init__(self, out_dim=192, act='gelu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, out_dim // 8, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim // 8), act_layer(act),
            nn.Conv2d(out_dim // 8, out_dim // 4, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim // 4), act_layer(act),
            nn.Conv2d(out_dim // 4, out_dim // 2, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim // 2), act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1), nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.convs(x)


class ViGTi(nn.Module):
    """Official Vision GNN ViG-Ti configuration compatible with vig_ti_74.5.pth."""
    def __init__(self, num_classes=3, drop_path_rate=0.0, drop_rate=0.0, num_knn=9):
        super().__init__()
        channels, blocks = 192, 12
        self.n_blocks = blocks
        self.stem = Stem(out_dim=channels, act='gelu')
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))
        dpr = torch.linspace(0, drop_path_rate, blocks).tolist()
        num_knn_per_block = torch.linspace(num_knn, 2 * num_knn, blocks).int().tolist()
        max_dilation = 196 // max(num_knn_per_block)
        self.backbone = nn.Sequential(*[
            nn.Sequential(
                Grapher(channels, num_knn_per_block[index], min(index // 4 + 1, max_dilation),
                        act='gelu', norm='batch', bias=True, stochastic=False, epsilon=0.2,
                        drop_path=dpr[index]),
                FFN(channels, channels * 4, act='gelu', drop_path=dpr[index]),
            ) for index in range(blocks)
        ])
        self.prediction = nn.Sequential(
            nn.Conv2d(channels, 1024, 1, bias=True), nn.BatchNorm2d(1024), act_layer('gelu'),
            nn.Dropout(drop_rate), nn.Conv2d(1024, num_classes, 1, bias=True),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


def vig_ti_224_gelu(num_classes=3, **kwargs):
    return ViGTi(num_classes=num_classes, **kwargs)


def load_vig_pretrained(model, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    state_dict = {(key[7:] if key.startswith('module.') else key): value for key, value in state_dict.items()}
    state_dict = {key: value for key, value in state_dict.items() if not key.startswith('prediction.4.')}
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith('prediction.4.')]
    missing = [key for key in incompatible.missing_keys if not key.startswith('prediction.4.')]
    if unexpected or missing:
        raise RuntimeError(f'ViG pretrained-weight mismatch; missing={missing}, unexpected={unexpected}')
    nn.init.trunc_normal_(model.prediction[4].weight, std=0.02)
    nn.init.zeros_(model.prediction[4].bias)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    random.seed(7 + worker_id)
    np.random.seed(7 + worker_id)
    torch.manual_seed(7 + worker_id)
    torch.cuda.manual_seed(7 + worker_id)
    torch.cuda.manual_seed_all(7 + worker_id)


if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(0)
    gpu_device = 0
    class_num = 3
    batch_size = 32
    epochs = 100
    image_size = [224, 224]
    image_dir = r'/root/autodl-tmp/GAFT/Datasets/Greece_Larynx'
    mode_stat = 'test'    # train, test, visual
    num_workers = 16
    pre_weight_path = '/root/autodl-tmp/Weights/ViG/vig_ti_74.5.pth'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/ViG_Greece_Larynx.pth'

    test_log_path = model_wegt_path.replace('.pth', '_training_log.txt')


    ########################## reading datas and processing datas #########################
    print('########################## reading datas and processing datas #########################')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224, 224]),
                                    transforms.Normalize(mean=0.5, std=0.5)])

    train_data = ImageFolder(image_dir + r'/Train_patch', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = ImageFolder(image_dir + r'/Val_patch', transform=transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_data = ImageFolder(image_dir + r'/Test_patch', transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ViG-Ti Backbone is defined in this file and matches vig_ti_74.5.pth.
    vig_net = vig_ti_224_gelu(num_classes=class_num)
    load_vig_pretrained(vig_net, pre_weight_path)
    summary(
        vig_net,
        input_size=(1, 3, 224, 224),
        depth=4
    )
    print(vig_net)
    vig_net = vig_net.cuda(gpu_device)

    if mode_stat == 'train':
        Single_out_fit(ddai_net=vig_net, train_loader=train_loader, val_loader=val_loader,
                       test_loader=test_loader, lr_fn='vit', epoch=epochs,
                       gpu_device=gpu_device, weight_path=model_wegt_path)

    elif mode_stat == 'test':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        vig_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=vig_net, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

    elif mode_stat == 'visual':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        vig_net.load_state_dict(larynx_weight, strict=True)
        features_3D = features_extraction_without_classes(test_model=vig_net, data_loader=test_loader,
                                                          test_model_name='vig')
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')


