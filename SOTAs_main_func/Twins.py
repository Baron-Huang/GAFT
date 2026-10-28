############################# twins_Demo ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing twins model

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
#from torchinfo import summary
from torchinfo import summary
from SIL_Utils.fit_functions import vit_lr_schedule, Single_out_fit, searching_best_lr, testing_funnction
from SIL_Utils.ablation_experiments import save_model, acc_scores, to_np_category
import cv2
from skimage import io
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from mmcls.models.backbones.twins import PCPVT, SVT
from SIL_Learning.Grad_CAM import Grad_CAM
from SOTAs_main_func.SIL_Learning.XGrad_CAM import XGrad_CAM
from SIL_Learning.Grad_CAM import Grad_CAM
import timm

class twins_Net(nn.Module):
    def __init__(self, class_num=3, base = None):
        super(twins_Net, self).__init__()
        self.base = base
        self.linear = nn.Linear(in_features=512, out_features=class_num)
        self.avrgp = nn.AvgPool2d(kernel_size=(7, 7))
        self.top_flat = nn.Flatten(start_dim=1, end_dim=3)

    def forward(self, x):
        y = self.base(x)
        y = y[0]
        y = self.avrgp(y)
        y = self.top_flat(y)
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
    setup_seed(0)
    gpu_device = 0
    class_num = 3
    batch_size = 32
    epochs = 100
    image_dir = r'/root/autodl-tmp/GAFT/Datasets/Greece_Larynx'
    mode_stat = 'test'    # train, test, visual
    num_workers = 16
    pre_weight_path = r'/root/autodl-tmp/Weights/Twins/twins-svt-small_3rdparty_8xb128_in1k_20220126-8fe5205b.pth'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/Twins_Greece_Larynx.pth'

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
    twins_base = SVT(arch='small')
    twins_base = timm.models.twins_svt_small(pretrained=False)
    weight = torch.load(pre_weight_path)
    weight = {k: v for k, v in weight.items() if (k in weight and 'head' not in k)}
    print(twins_base)
    """pre_weight_file = torch.load(
    r'E:/QYB/ViG_SwinT/ViG_SwinT_Weights/Twins/twins-svt-small_3rdparty_8xb128_in1k_20220126-8fe5205b.pth')
    twins_weight_file = twins_base.state_dict()
    pre_weight_keys = pre_weight_file.keys()
    twins_weight_keys = twins_weight_file.keys()
    pre_weight_list = list(pre_weight_keys)
    twins_weight_list = list(twins_weight_keys)
    search_len = len(twins_weight_list)
    print(pre_weight_list)
    print("#####################33")
    print(twins_weight_list)

    for i in range(search_len):
        if twins_weight_list[i] == pre_weight_list[i][9:]:
            print('success!!!')
            twins_weight_file[twins_weight_list[i]] = pre_weight_file['state_dict'][pre_weight_list[i]]

    twins_base.load_state_dict(twins_weight_file, strict=True)"""
    #twins_net = twins_Net(base=twins_base, class_num=class_num)
    twins_base.load_state_dict(weight, strict=False)
    twins_net = twins_base
    twins_net.eval()
    with torch.no_grad():
        print('########################## mlp_mixer_summary #########################')
        summary(
            twins_net,
            input_size=(1, 3, 224, 224),
            depth=4
        )
        #print('\n', '########################## mlp_mixer_net #########################')
        #print(twins_net, '\n')
    twins_net = twins_net.cuda(gpu_device)

    if mode_stat == 'train':
        Single_out_fit(ddai_net=twins_net, train_loader=train_loader, val_loader=val_loader,
                       test_loader=test_loader, lr_fn='vit', epoch=epochs,
                       gpu_device=gpu_device, weight_path=model_wegt_path)

    elif mode_stat == 'test':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        twins_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=twins_net, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

    elif mode_stat == 'visual':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        twins_net.load_state_dict(larynx_weight, strict=True)
        # print('########################## interpretable learning section #########################')
        #### ads-model
        # swinT_ads_cam_base = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=ads_cam_cls_num,
        #                                     embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        #                                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #                                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        #                                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        #                                     use_checkpoint=False, fused_window_process=False)
        # swinT_ads_cam_net = SwinT_Net(base_model=swinT_ads_cam_base, class_num=ads_cam_cls_num)

        # ads_weight = torch.load(r'E:/SOTA_Model_Interpretable_Learning/SIL_Weights/SwinT_ADS_CAM_Natural_mixed_ab.pth')
        # swinT_ads_cam_net.load_state_dict(ads_weight, strict=True)
        # swinT_ads_cam_net = swinT_ads_cam_net.cuda(gpu_device)

        print(isinstance(twins_net.norm, nn.BatchNorm2d))
        path_name = r'E:/QYB/ViG_SwinT/ViG_SwinT_Datasets/esophageal_cancer/Test_patch/I'
        img_name = r'455.jpg'
        img_name_list = os.listdir(path_name)
        image_path = path_name + img_name
        result_img_path = r'E:/QYB/ViG_SwinT/results/CAM_out(twins)/eso/image/I'
        result_cam_path = r'E:/QYB/ViG_SwinT/results/CAM_out(twins)/eso/cam/I'

        for i in img_name_list:
            ## swint, swinm
            my_xgrad_cam = XGrad_CAM(model=twins_net, path_name=path_name, model_name='twins', img_name=i,
                                     result_cam_path=result_cam_path, result_img_path=result_img_path,
                                     inverse_set=True, transform=transform, gpu_device=gpu_device, show_all_fp=True,
                                     training_img_path=None, map_size=[7, 7])

            my_xgrad_cam.get_all_result_images()
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')


