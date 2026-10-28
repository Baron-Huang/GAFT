############################# ConvNeXt_Demo ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing ConvNeXt model

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
#from SC_Model.Mixer_models.models.mixer import MLPMixer
from SIL_Utils.fit_functions import vit_lr_schedule, Single_out_fit,searching_best_lr, testing_funnction
from SIL_Utils.ablation_experiments import save_model, acc_scores, to_np_category
import cv2
from skimage import io
from mmcls.models.backbones.convnext import ConvNeXt
from SIL_Learning.Grad_CAM import Grad_CAM

class ConvNeXt_Net(nn.Module):
    def __init__(self, class_num=3, base = None):
        super(ConvNeXt_Net, self).__init__()
        self.base = base
        self.linear = nn.Linear(in_features=1024, out_features=class_num)

    def forward(self, x):
        y = self.base(x)
        y = y[0]
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
    mode_stat = 'test'    # train, test, visual
    image_dir = r'/root/autodl-tmp/GAFT/Datasets/Greece_Larynx'
    num_workers = 16
    pre_weight_path = r'/root/autodl-tmp/Weights/ConvNeXt/convnext-base_32xb128_in1k_20221207-fbdb5eb9.pth'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/ConvneXt_Greece_Larynx.pth'

    test_log_path = model_wegt_path.replace('.pth', '_training_log.txt')

    ########################## reading datas and processing datas #########################
    print('########################## reading datas and processing datas #########################')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224, 224]),
                                    transforms.Normalize(mean=0.5, std=0.5)])

    train_data = ImageFolder(image_dir + r'/Train_patch', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers= num_workers)
    val_data = ImageFolder(image_dir + r'/Val_patch', transform=transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers= num_workers)
    test_data = ImageFolder(image_dir + r'/Test_patch', transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers= num_workers)

    print('train_data:', '\n', train_data, '\n')

    ########################## creating models and visuling models #########################

    print('########################## creating models and visuling models #########################')
    convnext_base = ConvNeXt(arch='base')
    convnext_weight_file = convnext_base.state_dict()
    convnext_weight_keys = convnext_weight_file.keys()

    pre_weight_file = torch.load(pre_weight_path)
    pre_weight_keys = pre_weight_file['state_dict'].keys()

    pre_weight_list = list(pre_weight_keys)
    convnext_weight_list = list(convnext_weight_keys)
    search_len = len(convnext_weight_list)

    for i in range(search_len):
        if convnext_weight_list[i] == pre_weight_list[i][9:]:
            print('success!!!')
            convnext_weight_file[convnext_weight_list[i]] = pre_weight_file['state_dict'][pre_weight_list[i]]

    convnext_base.load_state_dict(convnext_weight_file, strict=True)
    convnext_net = ConvNeXt_Net(base=convnext_base)


    convnext_net.eval()
    with torch.no_grad():
        print('########################## convnext_summary #########################')
        summary(
            convnext_net,
            input_size=(1, 3, 224, 224),
            depth=4
        )
        print('\n', '########################## convnext_net #########################')
        print(convnext_net, '\n')
        print(convnext_net.base.stages[0][0].depthwise_conv.weight)
    convnext_net.cuda(gpu_device)

    if mode_stat == 'train':
        ########################## fitting models and testing models #########################
        print('########################## fitting models and testing models #########################')
        Single_out_fit(ddai_net=convnext_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                      lr_fn='vit', epoch = epochs, gpu_device = gpu_device, weight_path=model_wegt_path)

    elif mode_stat == 'test':
        ########################## testing function #########################
        print('########################## testing function #########################')
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        convnext_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model= convnext_net, train_loader=train_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

    elif mode_stat == 'visual':
        larynx_weight = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        convnext_net.load_state_dict(larynx_weight, strict=True)
        testing_funnction(test_model=convnext_net, train_loader=test_loader, val_loader=val_loader,
                          test_loader=test_loader, gpu_device=gpu_device, out_mode='single', weight_path=model_wegt_path, log_path=test_log_path)

        print(isinstance(convnext_net.base.stages[3][2].drop_path, nn.Identity))

        path_name = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_china\Test\III'
        img_name = r'455.jpg'
        img_name_list = os.listdir(path_name)
        image_path = path_name + img_name
        result_img_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_china\Grad_CAM\ConvNeXt\Images\III'
        result_cam_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_china\Grad_CAM\ConvNeXt\CAMs\III'

        for i in img_name_list:
            ## swint, swinm
            my_grad_cam = Grad_CAM(model=convnext_net, path_name=path_name, img_name=i, result_cam_path=result_cam_path,
                                   inverse_set=True, transform=transform, gpu_device=gpu_device, show_all_fp=True,
                                   training_img_path=None, result_img_path=result_img_path, map_size=[7, 7],
                                   model_name='convnext')

            my_grad_cam.get_all_result_images()
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')




