############################# ViT_Demo ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing Vinsion Transformer model

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
from SIL_Model.ViT_models.ViT import VisionTransformer, load_pretrained, vit_base_patch16_224, vit_small_patch16_224,\
    vit_base_patch32_224, vit_large_patch16_224, vit_large_patch32_224
from SIL_Model.ViT_models.ViT_model_modules import Dense_Net, ViT_Net
from SIL_Utils.fit_functions import  vit_lr_schedule, Single_out_fit, searching_best_lr, testing_funnction
from SIL_Utils.ablation_experiments import save_model, acc_scores, to_np_category
import cv2
from skimage import io
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from SIL_Learning.CAM import CAM
from SIL_Learning.Grad_CAM import Grad_CAM

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
    pre_weight_path = r'/root/autodl-tmp/Weights/ViT/jx_vit_base_p16_224-80ecf9dd.pth'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/ViT_Greece_Larynx.pth'

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
    vit_base = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=class_num, embed_dim=768,
                                            depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                            drop_rate=0.,
                                            attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
                                            norm_layer=nn.LayerNorm)

    vit_weight_file = torch.load(
        pre_weight_path)
    vit_weight_file = {k: v for k, v in vit_weight_file.items() if (k in vit_weight_file and 'head' not in k)}
    #vit_weight_file = {k: v for k, v in vit_weight_file.items() if
    #                   (k in vit_weight_file and 'patch_embed.proj' not in k)}
    vit_base.load_state_dict(vit_weight_file, strict=False)
    vit_net = ViT_Net(base_net=vit_base, class_num=class_num)
    #vit_net.load_state_dict(vit_weight_file, strict=False)
    nn.init.trunc_normal_(vit_net.head.weight)


    with torch.no_grad():
        print('########################## vit_summary #########################')
        summary(
            vit_net,
            input_size=(1, 3, 224, 224),
            depth=4
        )
        print('\n', '########################## vit_net #########################')
        print(vit_net, '\n')
    print(vit_net.transformer_block_0.mlp.fc1.weight)
    vit_net = vit_net.cuda(gpu_device)

    if mode_stat == 'train':
        Single_out_fit(ddai_net=vit_net, train_loader=train_loader, val_loader=val_loader,
                       test_loader=test_loader, lr_fn='vit', epoch=epochs,
                       gpu_device=gpu_device, weight_path=model_wegt_path)

    elif mode_stat == 'test':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        vit_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=vit_net, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

    elif mode_stat == 'visual':
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



        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        vit_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=vit_net, train_loader=test_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

        print(isinstance(vit_net.transformer_block_11.mlp.fc2, nn.Linear))

        path_name = r'E:/SOTA_Model_Interpretable_Learning/SIL_Datasets/Larynx_china/Test/III'
        img_name = r'455.jpg'
        img_name_list = os.listdir(path_name)
        image_path = path_name + img_name
        result_img_path = r'E:/SOTA_Model_Interpretable_Learning/SIL_Results/Larynx_china/Grad_CAM/ViT/Images/III'
        result_cam_path = r'E:/SOTA_Model_Interpretable_Learning/SIL_Results/Larynx_china/Grad_CAM/ViT/CAMs/III'

        for i in img_name_list:
            ## swint, swinm
            my_grad_cam = Grad_CAM(model=vit_net, path_name=path_name, img_name=i, result_cam_path=result_cam_path,
                                   inverse_set=True, transform=transform, gpu_device=gpu_device, show_all_fp=True,
                                   training_img_path=None, result_img_path=result_img_path, map_size=[14, 14],
                                   model_name='vit')

            my_grad_cam.get_all_result_images()
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')


