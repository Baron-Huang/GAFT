############################# Class Activation Mapping ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing Class Activation Mapping

########################## API Section #########################
import skimage.color
import torch
from torch import nn
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage import io, color, transform, metrics
import PIL
import seaborn as sns
from torchvision.transforms import InterpolationMode
sns.set(font='Times New Roman', font_scale=0.6)


class CAM:
    def __init__(self, model = None, path_name = None,  select_cls = 0, inverse_set = True,
                 transform = None, gpu_device = 0, show_all_fp = True, map_size = [7, 7],
                 training_img_path = None, img_name = None, result_img_path = None, result_cam_path = None):
        self.model = model
        self.select_cls = select_cls
        self.inverse_set = inverse_set
        self.path_name = path_name
        self.img_name = img_name
        self.transform = transform
        self.gpu_device = gpu_device
        self.show_all_fp = show_all_fp
        self.training_img_path = training_img_path
        self.transform_new = transforms.Resize([224, 224], interpolation=InterpolationMode.BILINEAR)
        self.result_img_path = result_img_path
        self.result_cam_path = result_cam_path
        self.map_size = [7, 7]

    def bese_function(self):
        finding_img_rgb = io.imread(self.path_name + r'\\' + self.img_name)
        finding_img = skimage.color.rgb2gray(finding_img_rgb)
        stat_img = finding_img.reshape(finding_img.shape[0] * finding_img.shape[1])

        avgp_weight = self.model.head.weight.data
        avgp_weight = avgp_weight[self.select_cls, :]
        max_num = torch.argmax(avgp_weight)

        show_fea = []

        def hook_fn(model, in_fea, out_fea):
            show_fea.append(out_fea)

        transform_img = transforms.Resize([224, 224], interpolation=InterpolationMode.NEAREST)
        vis_img = PIL.Image.open(self.path_name + r'\\' + self.img_name)
        vis_show_img = transform_img(vis_img)
        vis_show_img = np.array(vis_show_img)
        vis_show_img_gray = skimage.color.rgb2gray(vis_show_img)

        vis_img_tr = self.transform(vis_img)

        ##########
        self.model.norm.register_forward_hook(hook_fn)

        self.model.eval()
        with torch.no_grad():
            pre_y = self.model(vis_img_tr.view(1, 3, 224, 224).cuda(self.gpu_device))

        pre_cls_num = torch.argmax(nn.Softmax()(pre_y)).detach().cpu().numpy()
        pre_proba = nn.Softmax()(pre_y).detach().cpu().numpy()
        pre_proba = pre_proba[0, pre_cls_num]

        show_fea_np = show_fea[0].detach().cpu()
        show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], 768)

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                show_fea_np[i, j, :] = show_fea_np[i, j, :] * avgp_weight.detach().cpu()
        fea_map = show_fea_np * 1

        fea_map_show = torch.mean(fea_map, dim=2)
        fea_map_show = (fea_map_show - torch.mean(fea_map_show)) / torch.std(fea_map_show)
        fea_map_show = fea_map_show - torch.min(fea_map_show)

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if fea_map_show[i, j] <=0:
                    fea_map_show[i, j] = 0

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if fea_map_show[i, j] <= 0.8 * torch.max(fea_map_show):
                    fea_map_show[i, j] = 0

        fea_map_show = fea_map_show / torch.max(fea_map_show)

        transform_map = transforms.Resize([224, 224], interpolation=InterpolationMode.BILINEAR)
        fea_map_show = transform_map(fea_map_show.reshape(1, self.map_size[0], self.map_size[1]))  ## CAM
        fea_map_show = fea_map_show.permute(1, 2, 0).reshape(224, 224)
        fea_map_show = fea_map_show.numpy()
        xxxx = fea_map_show + 0

        fea_map_show = np.uint8(fea_map_show * 255.0)

        if self.inverse_set == True:
            fea_map_show = 255 - fea_map_show
        else:
            pass

        fea_map_show = cv2.applyColorMap(fea_map_show, cv2.COLORMAP_JET)
        weighted_image = cv2.addWeighted(vis_show_img, 0.7, fea_map_show, 0.4, 0.3)

        del self.model

        return (vis_show_img_gray, finding_img, stat_img, vis_show_img, xxxx, fea_map_show, weighted_image,
                 show_fea_np, pre_proba, pre_cls_num, vis_img_tr, finding_img_rgb)

    ########################## show_results #########################
    def show_results(self):
        (vis_show_img_gray, finding_img, stat_img, vis_show_img, xxxx, fea_map_show, weighted_image,
       show_fea_np, pre_proba, pre_cls_num, vis_img_tr, finding_img_rgb) \
            = self.bese_function()

        plt.figure(1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(finding_img, cmap='gray')

        plt.figure(2)
        sns.histplot(stat_img, bins=5, color='black', stat='density', kde=True)

        plt.figure(3)
        plt.subplot(2, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(vis_show_img)

        plt.subplot(2, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(xxxx)

        plt.subplot(2, 2, 3)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(fea_map_show)

        plt.subplot(2, 2, 4)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(weighted_image)

        #io.imsave('E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\\New_datasets\Single\show_img.jpg',
        #          weighted_image)

        if self.show_all_fp == True:
            plt.figure(4, figsize=(12, 16))
            for i in range(24):
                for j in range(32):
                    plt.subplot(24, 32, (32 * i + j) + 1)
                    plt.xticks([])
                    plt.yticks([])
                    if i == 0:
                        plt.title(j)
                    if j == 0:
                        plt.ylabel(i * 32)
                    plt.imshow(show_fea_np[:, :, (32 * i + j)])


        plt.show()

        print('predict results:', pre_cls_num)
        print('predict probability:', pre_proba)

    ########################## get_result_images #########################
    def get_all_result_images(self):
        (vis_show_img_gray, finding_img, stat_img, vis_show_img, xxxx, fea_map_show, weighted_image,
         show_fea_np, pre_proba, pre_cls_num, vis_img_tr, finding_img_rgb) \
            = self.bese_function()

        io.imsave(self.result_img_path + r'\\' + self.img_name, weighted_image)
        io.imsave(self.result_cam_path + r'\\' + self.img_name, fea_map_show)
        print(self.img_name)