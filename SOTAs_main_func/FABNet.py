############################# ViT_Demo ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing FABNet model

########################## API Section #########################
import torch
from torch import nn
from torchvision.models.densenet import densenet121
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torchinfo import summary
from SIL_Model.FABNet_models.FABNet_model_modules import FABNet, Dense_Net
from SIL_Utils.fit_functions import Single_out_fit, testing_funnction
import cv2
from skimage import io
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
#from SIL_Learning.Adaptive_Dual_Semantic_CAM import CAM

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

if __name__ == '__main__':
    ########################## Hyparameters #########################
    setup_seed(0)
    gpu_device = 0
    class_num = 3
    #ads_cam_cls_num = 2
    batch_size = 32
    epochs = 100
    image_dir = r'/root/autodl-tmp/GAFT/Datasets/Greece_Larynx'
    mode_stat = 'test'    # train, test, visual
    num_workers = 16
    pre_weight_path = r'/root/autodl-tmp/Weights/DenseNet121/densenet121-a639ec97.pth'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/FABNet_Greece_Larynx.pth'

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

    ########################## creating a FABNet model #########################
    print('########################## creating a FABNet model #########################')
    dense_base = densenet121(pretrained=False)
    dense_weight_file = torch.load(pre_weight_path)
    dense_base.load_state_dict(dense_weight_file, strict=False)
    fab_net = FABNet(base_net=dense_base, class_num=class_num)
    summary(
        fab_net,
        input_size=(1, 3, 224, 224),
        depth=4
    )
    print(fab_net)
    fab_net = fab_net.cuda(gpu_device)

    if mode_stat == 'train':
        Single_out_fit(ddai_net=fab_net, train_loader=train_loader, val_loader=val_loader,
                       test_loader=test_loader, lr_fn='vit', epoch=epochs,
                       gpu_device=gpu_device, weight_path=model_wegt_path)

    elif mode_stat == 'test':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        fab_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=fab_net, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

    elif mode_stat == 'visual':
        #print('########################## interpretable learning section #########################')
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        fab_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=fab_net, train_loader=train_loader, val_loader=val_loader,
                        test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

        #print(isinstance(fab_net.top_AvgMp, nn.AvgPool2d))
        # print(swinT_net.norm)
        # print(swinT_net.avgp)

        path_name = r'E:/SOTA_Model_Interpretable_Learning/SIL_Datasets/New_datasets/Single//'
        img_name_list = os.listdir(path_name)

        training_img_path = r'E:/SOTA_Model_Interpretable_Learning/SIL_Results/ST_MSLNet//1-All//'
        result_img_path = r'E:/SOTA_Model_Interpretable_Learning/SIL_Results/ST_MSLNet//2-Test//'

        for i in img_name_list:
            my_ads_cam = CAM(model=fab_net, path_name=path_name, img_name=i,
                             select_cls=0, inverse_set=True,
                             show_all_fp=True, rd_setting=False,
                             transform=transform, training_img_path=training_img_path,
                             result_img_path=result_img_path)
        my_ads_cam.show_results()
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')


