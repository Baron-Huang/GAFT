#!/root/miniconda3/envs/aloe/bin/python /root/autodl-tmp/GAFT/Codes/SOTAs_main_func/ALOE.py

import random
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


# transformers>=4.56 calls the public pytree API introduced by PyTorch 2.2.
# Keep the separate ALOE environment compatible with PyTorch 2.1 by adapting
# its private equivalent and ignoring metadata unsupported by that release.
_pytree = torch.utils._pytree
if not hasattr(_pytree, 'register_pytree_node') and hasattr(_pytree, '_register_pytree_node'):
    def _register_pytree_node_compat(node_type, flatten_fn, unflatten_fn, **_kwargs):
        return _pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

    _pytree.register_pytree_node = _register_pytree_node_compat

from torch import nn
# Make the repository's Codes directory importable when PyCharm runs this
# script directly from SOTAs_main_func.
_codes_root = str(Path(__file__).resolve().parent.parent)
if _codes_root not in sys.path:
    sys.path.insert(0, _codes_root)

from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import ImageFolder

from SIL_Utils.fit_functions import Single_out_fit, testing_funnction


class ALOEClassifier(nn.Module):
    """ALOE DINOv3-Base (ViT-B/16) backbone with a GAFT_SwinT_Greece_Larynx classifier head."""
    def __init__(self, pretrained_path, num_classes=3):
        super().__init__()
        model_dir = Path(pretrained_path).resolve()
        required_files = ('__init__.py', 'configuration_aloe_vision.py', 'modeling_aloe_dinov3.py')
        missing_files = [name for name in required_files if not (model_dir / name).is_file()]
        if missing_files:
            raise FileNotFoundError(f'ALOE 模型目录缺少文件: {missing_files}; path={model_dir}')

        # Load the complete local ALOE runtime as a package. This avoids the
        # Transformers dynamic-module cache and its online remote-code lookup.
        package_name = 'local_aloe_arch'
        if package_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                package_name,
                model_dir / '__init__.py',
                submodule_search_locations=[str(model_dir)],
            )
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            spec.loader.exec_module(package)
        model_module = importlib.import_module(f'{package_name}.modeling_aloe_dinov3')
        self.backbone = model_module.AloeDinoV3VisionModel.from_pretrained(str(model_dir))
        hidden_size = getattr(self.backbone.config, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, 'embed_dim', None)
        if hidden_size is None:
            raise AttributeError('无法从 ALOE 配置读取 Backbone 特征维度。')
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # ALOE 的 B-cos stem 使用六通道输入：RGB 及其颜色互补通道。
        # 数据加载器已将 RGB 以 mean=std=0.5 归一化到 [-1, 1]，因此
        # RGB 的互补色经过相同归一化后恰好为 -x。
        expected_channels = getattr(self.backbone.config, 'num_channels', 6)
        if x.shape[1] == 3 and expected_channels == 6:
            x = torch.cat((x, -x), dim=1)
        elif x.shape[1] != expected_channels:
            raise ValueError(
                f'ALOE Backbone 需要 {expected_channels} 通道输入，当前输入为 {x.shape[1]} 通道。'
            )

        outputs = self.backbone(pixel_values=x)
        if getattr(outputs, 'pooler_output', None) is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.classifier(features)


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
    pre_weight_path = r'/root/autodl-tmp/Weights/ALOE/aloe-dinov3-base-in1k-lp'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/ALOE_Greece_Larynx.pth'

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

    aloe_net = ALOEClassifier(pretrained_path=pre_weight_path, num_classes=class_num)
    print(aloe_net)
    aloe_net = aloe_net.cuda(gpu_device)

    if mode_stat == 'train':
        Single_out_fit(ddai_net=aloe_net, train_loader=train_loader, val_loader=val_loader,
                       test_loader=test_loader, lr_fn='vit', epoch=epochs,
                       gpu_device=gpu_device, weight_path=model_wegt_path)
    elif mode_stat == 'test':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        aloe_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=aloe_net, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)
    elif mode_stat == 'visual':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        aloe_net.load_state_dict(larynx_weight, strict=True)
        features_extraction_without_classes(test_model=aloe_net, data_loader=test_loader,
                                            test_model_name='ALOE')
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')


