############################# DeiT_Demo ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing DeiT model

########################## API Section #########################
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torchinfo import summary
from SIL_Utils.fit_functions import vit_lr_schedule, Single_out_fit, searching_best_lr, testing_funnction
from SIL_Utils.fit_functions import vit_lr_schedule, Single_out_fit,searching_best_lr
from SIL_Utils.ablation_experiments import save_model, acc_scores, to_np_category
import cv2
from skimage import io
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from mmcls.models.backbones.deit import DistilledVisionTransformer
from mmcls.models.backbones.twins import PCPVT

class DeiT_Net(nn.Module):
    def __init__(self, class_num=3, base = None):
        super(DeiT_Net, self).__init__()
        self.base = base
        self.linear = nn.Linear(in_features=768, out_features=class_num)

    def forward(self, x):
        y = self.base(x)
        y = y[0][2]
        y = self.linear(y)
        return y

########################## seed_function #########################
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

########################## main_function #########################
if __name__ == '__main__':
    ########################## Hyparameters #########################
    setup_seed(0)
    gpu_device = 0
    class_num = 3
    batch_size = 32
    epochs = 100
    image_dir = r'/root/autodl-tmp/GAFT/Datasets/Greece_Larynx'
    mode_stat = 'test'    # train, test, visual
    num_workers = 16
    pre_weight_path = r'/root/autodl-tmp/Weights/DeiT/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/DeiT_Greece_Larynx.pth'

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

    print('train_data:', '\n', train_data, '\n')

    ########################## creating models and visuling models #########################

    print('########################## creating models and visuling models #########################')
    deit_base = DistilledVisionTransformer(arch='base', img_size=224)
    pre_weight_file = torch.load(
        pre_weight_path)
    deit_weight_file = deit_base.state_dict()
    pre_weight_keys = pre_weight_file['state_dict'].keys()
    deit_weight_keys = deit_weight_file.keys()
    pre_weight_list = list(pre_weight_keys)
    deit_weight_list = list(deit_weight_keys)
    print(pre_weight_list)
    print('################################################')
    print(deit_weight_list)
    search_len = len(deit_weight_list)
    deit_weight_file[deit_weight_list[0]] = pre_weight_file['state_dict'][pre_weight_list[0]]
    i = 2
    for i in range(search_len):
        if deit_weight_list[i] == pre_weight_list[i-1][9:]:# and deit_weight_list[i] != deit_weight_list[1]:
            print('success!!!')
            deit_weight_file[deit_weight_list[i]] = pre_weight_file['state_dict'][pre_weight_list[i-1]]

    deit_base.load_state_dict(deit_weight_file, strict=True)
    deit_net = DeiT_Net(base=deit_base, class_num=class_num)
    deit_net.eval()
    with torch.no_grad():
        print('########################## mlp_mixer_summary #########################')
        summary(
            deit_net,
            input_size=(1, 3, 224, 224),
            depth=4
        )
        print('\n', '########################## mlp_mixer_net #########################')
        print(deit_net, '\n')
        print(deit_net.base.ln1.weight)
    deit_net = deit_net.cuda(gpu_device)

    if mode_stat == 'train':
        Single_out_fit(ddai_net=deit_net, train_loader=train_loader, val_loader=val_loader,
                       test_loader=test_loader, lr_fn='vit', epoch=epochs,
                       gpu_device=gpu_device, weight_path=model_wegt_path)

    elif mode_stat == 'test':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        deit_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=deit_net, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

    elif mode_stat == 'visual':
        raise NotImplementedError('This demo has no visualization implementation.')
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')


