import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import ImageFolder
from timm.models.swin_transformer import SwinTransformer

from SIL_Utils.fit_functions import Single_out_fit, testing_funnction


class ThinkingViTSwinSmall(SwinTransformer):
    """ThinkingViT-Swin Backbone using the official Swin-S full second-stage capacity.

    The official ThinkingViT-Swin schedule is (3, 3, 6, 12) -> (3, 6, 12, 24).
    This downstream classifier uses the full second stage, whose parameter layout is
    compatible with the published ThinkingViT-Swin checkpoint.
    """
    def __init__(self, num_classes=3):
        super().__init__(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )


def thinkingvit_swin_small(num_classes=3):
    return ThinkingViTSwinSmall(num_classes=num_classes)


def load_thinkingvit_pretrained(model, weight_path):
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError('缺少 safetensors；请运行: python -m pip install safetensors') from error
    state_dict = load_file(weight_path, device='cpu')
    state_dict = {(key[7:] if key.startswith('module.') else key): value for key, value in state_dict.items()}
    ignored_prefixes = ('head.', 'alpha.', 'proj_layer.')
    state_dict = {
        key: value for key, value in state_dict.items()
        if not key.startswith(ignored_prefixes)
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith('head.')]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith(ignored_prefixes)]
    if missing or unexpected:
        raise RuntimeError(f'ThinkingViT-Swin 权重不匹配：missing={missing}, unexpected={unexpected}')
    #torch.nn.init.trunc_normal_(model.head.weight, std=0.02)
    #if model.head.bias is not None:
    #    torch.nn.init.zeros_(model.head.bias)


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
    setup_seed(0)
    gpu_device = 0
    class_num = 3
    batch_size = 32
    epochs = 100
    image_size = [224, 224]
    image_dir = r'/root/autodl-tmp/GAFT/Datasets/Greece_Larynx'
    mode_stat = 'test'  # train, test, visual
    num_workers = 16
    pre_weight_path = r'/root/autodl-tmp/Weights/ThinkingViT/thinkingvit-swin-s-imagenet1k.safetensors'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/ThinkingViT_Greece_Larynx.pth'

    test_log_path = model_wegt_path.replace('.pth', '_training_log.txt')

    print('########################## reading datas and processing datas #########################')
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize([224, 224]), transforms.Normalize(mean=0.5, std=0.5)
    ])
    train_data = ImageFolder(image_dir + r'/Train_patch', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = ImageFolder(image_dir + r'/Val_patch', transform=transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_data = ImageFolder(image_dir + r'/Test_patch', transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    thinkingvit_net = thinkingvit_swin_small(num_classes=class_num)
    load_thinkingvit_pretrained(thinkingvit_net, pre_weight_path)
    summary(thinkingvit_net, input_size=(1, 3, 224, 224), depth=4)
    print(thinkingvit_net)
    thinkingvit_net = thinkingvit_net.cuda(gpu_device)

    if mode_stat == 'train':
        Single_out_fit(ddai_net=thinkingvit_net, train_loader=train_loader, val_loader=val_loader,
                       test_loader=test_loader, lr_fn='vit', epoch=epochs,
                       gpu_device=gpu_device, weight_path=model_wegt_path)
    elif mode_stat == 'test':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        thinkingvit_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=thinkingvit_net, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)
    elif mode_stat == 'visual':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        thinkingvit_net.load_state_dict(larynx_weight, strict=True)
        features_extraction_without_classes(test_model=thinkingvit_net, data_loader=test_loader,
                                            test_model_name='ThinkingViT')
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')

