"""SwinT + ViG feature-level weighted-summation baseline for AMU_LSCC_ROI."""
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

CODES_ROOT = Path(__file__).resolve().parents[1]
if str(CODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CODES_ROOT))

from models.gcn import vig_ti_224_gelu
from models.my_net import Weighted_Summation
from models.swin_transformer import swin_tiny
from utils.fit_functions import testing_fusion_function, train_weighted_summation


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    random.seed(7 + worker_id)
    np.random.seed(7 + worker_id)
    torch.manual_seed(7 + worker_id)


def load_pretrained_backbones(class_num, swin_path, vig_path):
    swin_base = swin_tiny(class_num=class_num)
    swin_checkpoint = torch.load(swin_path, map_location='cpu')
    swin_state = swin_checkpoint.get('model', swin_checkpoint)
    swin_state = {key: value for key, value in swin_state.items() if not key.startswith('head.')}
    swin_base.load_state_dict(swin_state, strict=False)
    nn.init.trunc_normal_(swin_base.head.weight, std=.02)
    nn.init.zeros_(swin_base.head.bias)

    vig_base = vig_ti_224_gelu(class_num=class_num)
    vig_checkpoint = torch.load(vig_path, map_location='cpu')
    vig_state_raw = vig_checkpoint.get('model', vig_checkpoint.get('state_dict', vig_checkpoint))
    vig_state = {}
    for key, value in vig_state_raw.items():
        key = key[7:] if key.startswith('module.') else key
        if not key.startswith('prediction.4'):
            vig_state[key] = value
    vig_base.load_state_dict(vig_state, strict=False)
    nn.init.trunc_normal_(vig_base.prediction[4].weight, std=.02)
    nn.init.zeros_(vig_base.prediction[4].bias)
    return swin_base, vig_base


if __name__ == '__main__':
    setup_seed(0)
    class_num = 3
    batch_size = 32
    epochs = 100
    image_size = 224
    num_workers = 16
    data_root = r'/root/autodl-tmp/GAFT/Datasets/AMU_LSCC_ROI'
    swinT_path = r'/root/autodl-tmp/GAFT/Results/weight/swin_tiny_patch4_window7_224_22k.pth'
    gcn_path = r'/root/autodl-tmp/GAFT/Results/weight/vig_ti_74.5.pth'
    model_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Fusion_Methods/weighted_summation.pth'
    model_stat = 'test'  # train, test
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([image_size, image_size], antialias=True),
        transforms.Normalize(mean=0.5, std=0.5),
    ])
    train_loader = DataLoader(ImageFolder(data_root + r'/Train_patch', transform=transform),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(ImageFolder(data_root + r'/Val_patch', transform=transform),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ImageFolder(data_root + r'/Test_patch', transform=transform),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers)

    swin_base, vig_base = load_pretrained_backbones(class_num, swinT_path, gcn_path)
    fusion_net = Weighted_Summation(swinT_base=swin_base, vig_base=vig_base, class_num=class_num)

    if model_stat == 'train':
        train_weighted_summation(model=fusion_net, train_loader=train_loader, val_loader=val_loader,
                                 test_loader=test_loader, epoch=epochs, gpu_device=device,
                                 weight_path=model_path)
    elif model_stat == 'test':
        fusion_net.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        fusion_net = fusion_net.to(device)
        testing_fusion_function(test_model=fusion_net, train_loader=train_loader, val_loader=val_loader,
                                test_loader=test_loader, gpu_device=0, out_mode='triplet', weight_path=model_path)
    else:
        raise ValueError(f'Unsupported model_stat: {model_stat!r}')
