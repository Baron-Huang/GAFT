############################# ResNeXt_Demo ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing ResNeXt model

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
from SIL_Utils.fit_functions import vit_lr_schedule, Single_out_fit,searching_best_lr, testing_funnction
from SIL_Utils.ablation_experiments import save_model, acc_scores, to_np_category
import cv2
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from mmcls.models.backbones.resnext import ResNeXt
from SIL_Learning.Grad_CAM import Grad_CAM
from utils.train_utils import CapturePrint, PrintCapture
import contextlib
from feature_map_out import feature_out

class ResNeXt_Net(nn.Module):
    def __init__(self, class_num=3, base = None):
        super(ResNeXt_Net, self).__init__()
        self.base = base
        self.linear = nn.Linear(in_features=2048, out_features=class_num)
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
    ########################## Hyparameters #########################
    setup_seed(0)
    gpu_device = 0
    class_num = 3
    batch_size = 32
    epochs = 100
    image_dir = r'/root/autodl-tmp/GAFT/Datasets/Greece_Larynx'
    mode_stat = 'test'    # train, test, visual
    num_workers = 16
    pre_weight_path = r'/root/autodl-tmp/Weights/ResNeXt/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/ResNeXt_Greece_Larynx.pth'

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
    resnext_base = ResNeXt(depth = 50)
    pre_weight_file = torch.load(
        pre_weight_path,map_location=f'cuda:{gpu_device}')
    resnext_weight_file = resnext_base.state_dict()
    pre_weight_keys = pre_weight_file['state_dict'].keys()
    resnext_weight_keys = resnext_weight_file.keys()
    pre_weight_list = list(pre_weight_keys)
    resnext_weight_list = list(resnext_weight_keys)
    search_len = len(resnext_weight_list)
    target_len = len(pre_weight_list)

    for i in range(search_len):
        for j in range(target_len):
            if resnext_weight_list[i] == pre_weight_list[j][9:]:
                print('success!!!')
                resnext_weight_file[resnext_weight_list[i]] = pre_weight_file['state_dict'][pre_weight_list[j]]

    resnext_base.load_state_dict(resnext_weight_file, strict=True)
    resnext_net = ResNeXt_Net(base=resnext_base, class_num=class_num)

    resnext_net.eval()
    with torch.no_grad():
        print('########################## mlp_mixer_summary #########################')
        summary(
            resnext_net,
            input_size=(1, 3, 224, 224),
            depth=4
        )
        print('\n', '########################## mlp_mixer_net #########################')
        print(resnext_net, '\n')
    resnext_net = resnext_net.cuda(gpu_device)

    if mode_stat == 'train':
        Single_out_fit(ddai_net=resnext_net, train_loader=train_loader, val_loader=val_loader,
                       test_loader=test_loader, lr_fn='vit', epoch=epochs,
                       gpu_device=gpu_device, weight_path=model_wegt_path)

    elif mode_stat == 'test':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        resnext_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=resnext_net, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

    elif mode_stat == 'visual':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        resnext_net.load_state_dict(larynx_weight, strict=True)
        path_name = r'/data/QYB/GMamba/dataset/Larynx_datasets_ST_MSLNet_Random/Test_patch/III'
        img_name_list = os.listdir(path_name)
        img_name = r'79.jpg'
        image_path = path_name + img_name
        result_feature_map_path = r'/data/QYB/ViT2GNN/featuremap_picture/resnext/lar/III'
        block_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for img_name in img_name_list:
            for idx, i in enumerate(block_select):
                my_feature = feature_out(model=resnext_net, model_name='resnext', out_model='single',
                                         i=i, transform=transform, path_name=path_name, img_name=img_name,
                                         result_feature_map_path=result_feature_map_path)
                my_feature.bese_function()
        ########################## fitting models and testing models #########################
        # print('########################## fitting models and testing models #########################')
        # weight_save_path = f'/data/QYB/GMamba/result/contrast/ResNeXt50/resnext_bre.pth'
        # log_file_path = f'/data/QYB/GMamba/result/contrast/ResNeXt50/resnext_bre.txt'
        #
        # capture_1 = CapturePrint()
        # capture_2 = CapturePrint()
        # print_capture_1 = PrintCapture(capture_1)
        # print_capture_2 = PrintCapture(capture_2)
        # with contextlib.redirect_stdout(print_capture_1) as f:
        #     Single_out_fit(ddai_net=resnext_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        #                    lr_fn='vit', epoch=epochs, gpu_device=gpu_device,
        #                    weight_path=weight_save_path)
        #     train_logs = f.getvalue()
        # with open(log_file_path, 'w') as f:
        #     f.write('training logs:\n')
        #     f.write(train_logs)
        #
        # with contextlib.redirect_stdout(print_capture_2) as f:
        #     testing_funnction(test_model=resnext_net, train_loader=train_loader, val_loader=val_loader,
        #                       test_loader=test_loader,
        #                       gpu_device=gpu_device, out_mode='single')
        #     test_logs = f.getvalue()
        # with open(log_file_path, 'a') as f:
        #     f.write('\ntesting logs:\n')
        #     f.write(test_logs)

        #Single_out_fit(ddai_net=resnext_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                       #lr_fn='vit', epoch=epochs, gpu_device=gpu_device,
                       #weight_path=r'E:/QYB/ViG_SwinT/results/contrast/ResNeXt50/resnext_eso.pth')

        ########################## searching the best learning rate #########################
        #print('########################## searching the best learning rate #########################')
        #searching_best_lr(min_lr=1e-7, max_lr=1e-4, fitting_model='single_out',
        #                  train_loader=train_loader, val_loader=val_loader, class_num=class_num,
        #                  test_loader=test_loader, lr_fn='searching_best_lr', epoch=epochs, gpu_device=0,
        #                  onecycle_mr=1e-5)

        ########################## testing function #########################
        #print('########################## testing function #########################')
        #larynx_weight = torch.load(
        #    r'E:/SOTA_Model_Interpretable_Learning/SIL_Weights/Larynx_china/resnext-Larynx-china-2080ti-r1.pth',
        #    map_location=f'cuda:{gpu_device}')
        #resnext_net.load_state_dict(larynx_weight, strict=True)
        #testing_funnction(test_model=resnext_net, train_loader=train_loader, val_loader=val_loader,
        #                  test_loader=test_loader, gpu_device=gpu_device, out_mode='single')

        ########################## interpretable learning section #########################
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

        # larynx_weight = torch.load(
        #     r'/data/QYB/GMamba/result/contrast/ResNeXt50/resnext_eso.pth',
        # map_location=f'cuda:{gpu_device}')
        # resnext_net.load_state_dict(larynx_weight, strict=True)
        # testing_funnction(test_model=resnext_net, train_loader=test_loader, val_loader=val_loader,
        #                   test_loader=test_loader, gpu_device=gpu_device, out_mode='single')

        # print(isinstance(resnext_net.base.layer4[1].conv3, nn.Conv2d))
        #
        # path_name = r'E:/SOTA_Model_Interpretable_Learning/SIL_Datasets/Larynx_china/Test/III'
        # img_name = r'455.jpg'
        # img_name_list = os.listdir(path_name)
        # image_path = path_name + img_name
        # training_img_path = r'E:/SOTA_Model_Interpretable_Learning/SIL_Results/ST_MSLNet//1-All'
        # result_img_path = r'E:/SOTA_Model_Interpretable_Learning/SIL_Results/Larynx_china/Grad_CAM/ResNeXt/Images/III'
        # result_cam_path = r'E:/SOTA_Model_Interpretable_Learning/SIL_Results/Larynx_china/Grad_CAM/ResNeXt/CAMs/III'
        #
        # for i in img_name_list:
        #     my_grad_cam = Grad_CAM(model = resnext_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
        #             inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
        #              training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7],
        #                            model_name = 'resnext')
        #
        #     my_grad_cam.get_all_result_images()
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')


