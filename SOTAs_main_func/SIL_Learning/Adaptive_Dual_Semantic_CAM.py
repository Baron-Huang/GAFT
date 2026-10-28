############################# Adaptive_Dual_Semantic_CAM ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing Adaptive Dual Semantic-CAM

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

class Adaptive_Dual_Semantic_CAM:
    def __init__(self, model = None, path_name = None, select_cls = 0, inverse_set = True, set_rate_1 = 0.55,
                 set_rate_2 = 0.55, rd_setting = True, transform = None, gpu_device = 0, show_all_fp = True,
                 training_img_path = None, img_name = None, ads_cam_model = None, result_img_path = None,
                 result_cam_path = None, adaptive_strategy = 'histopathology', out_mode = 'single'):
        self.model = model
        self.select_cls = select_cls
        self.inverse_set = inverse_set
        self.set_rate_1 = set_rate_1
        self.set_rate_2 = set_rate_2
        self.rd_setting = rd_setting
        self.path_name = path_name + r'\\'
        self.img_name = img_name
        self.transform = transform
        self.gpu_device = gpu_device
        self.show_all_fp = show_all_fp
        self.training_img_path = training_img_path
        self.ads_cam_model = ads_cam_model
        self.transform_new = transforms.Resize([224, 224], interpolation=InterpolationMode.BILINEAR)
        self.adaptive_strategy = adaptive_strategy
        self.result_img_path = result_img_path
        self.result_cam_path = result_cam_path
        self.out_mode = out_mode

    def bese_function(self):
        finding_img_rgb = io.imread(self.path_name + self.img_name)
        finding_img = skimage.color.rgb2gray(finding_img_rgb)
        stat_img = finding_img.reshape(finding_img.shape[0] * finding_img.shape[1])

        avgp_weight = self.model.head.weight.data
        avgp_weight = avgp_weight[self.select_cls, :]
        max_num = torch.argmax(avgp_weight)
        max_num = max_num.detach().cpu().numpy()

        show_fea = []
        test_num = max_num
        rd_setting = self.rd_setting

        def hook_fn(model, in_fea, out_fea):
            if rd_setting == True:
                out_fea[:, :, :test_num] = torch.zeros_like(out_fea[:, :, :test_num], device=out_fea.device)
                out_fea[:, :, (test_num + 1):] = torch.zeros_like(out_fea[:, :, (test_num) + 1:], device=out_fea.device)

            show_fea.append(out_fea)

        transform_img = transforms.Resize([224, 224], interpolation=InterpolationMode.NEAREST)
        vis_img = PIL.Image.open(self.path_name + self.img_name)
        vis_show_img = transform_img(vis_img)
        vis_show_img = np.array(vis_show_img)
        vis_show_img_gray = skimage.color.rgb2gray(vis_show_img)

        vis_img_tr = self.transform(vis_img)

        #
        self.model.norm.register_forward_hook(hook_fn)

        self.model.eval()
        with torch.no_grad():
            if self.out_mode == 'single':
                pre_y = self.model(vis_img_tr.view(1, 3, 224, 224).cuda(self.gpu_device))
            elif self.out_mode == 'triplet':
                _, _, pre_y = self.model(vis_img_tr.view(1, 3, 224, 224).cuda(self.gpu_device))

        pre_cls_num = torch.argmax(nn.Softmax()(pre_y)).detach().cpu().numpy()
        pre_proba = nn.Softmax()(pre_y).detach().cpu().numpy()
        pre_proba = pre_proba[0, pre_cls_num]

        show_fea_np = show_fea[0].detach().cpu().reshape(49, 768)
        show_fea_np = show_fea_np.reshape(7, 7, 768)
        fea_map = show_fea_np * 1

        for s in range(7):
            for k in range(7):
                fea_map[s, k] = fea_map[s, k] * avgp_weight.detach().cpu()

        fea_map_show = torch.mean(fea_map, dim=2)
        fea_map_show[0, 0] = torch.mean(fea_map_show)
        fea_map_show[0, 6] = torch.mean(fea_map_show)
        fea_map_show[6, 0] = torch.mean(fea_map_show)
        fea_map_show[6, 6] = torch.mean(fea_map_show)
        new_fea_map_show = fea_map_show * 1

        max_value = torch.max(fea_map_show)

        fea_map_show = max_value - fea_map_show

        max_value = torch.max(fea_map_show)

        for s in range(7):
            for k in range(7):
                if (fea_map_show[s, k] < max_value * self.set_rate_1):
                    fea_map_show[s, k] = torch.tensor(0)

        transform_map = transforms.Resize([224, 224], interpolation=InterpolationMode.BILINEAR)
        fea_map_show = transform_map(fea_map_show.reshape(1, 7, 7))  ## CAM
        fea_map_show = fea_map_show.permute(1, 2, 0).reshape(224, 224)
        fea_map_show = fea_map_show.numpy()
        xxxx = fea_map_show + 0

        if np.min(fea_map_show) < 0:
            fea_map_show = fea_map_show + (0 - np.min(fea_map_show))
            fea_map_show = (fea_map_show - np.min(fea_map_show)) / np.max(fea_map_show)
        else:
            fea_map_show = (fea_map_show - np.min(fea_map_show)) / np.max(fea_map_show)

        fea_map_show = np.uint8(fea_map_show * 255.0)

        if self.inverse_set == True:
            fea_map_show = 255 - fea_map_show
        else:
            pass

        fea_map_show = cv2.applyColorMap(fea_map_show, cv2.COLORMAP_JET)
        weighted_image = cv2.addWeighted(vis_show_img, 0.8, fea_map_show, 0.4, 0.3)

        ####################################################################################
        new_max_value = torch.max(new_fea_map_show)

        for s in range(7):
            for k in range(7):
                if (new_fea_map_show[s, k] < new_max_value * self.set_rate_2):
                    new_fea_map_show[s, k] = torch.tensor(0)

        new_fea_map_show = transform_map(new_fea_map_show.reshape(1, 7, 7))  ## CAM
        new_fea_map_show = new_fea_map_show.permute(1, 2, 0).reshape(224, 224)
        new_fea_map_show = new_fea_map_show.numpy()
        ssss = new_fea_map_show

        if np.min(new_fea_map_show) < 0:
            new_fea_map_show = new_fea_map_show + (0 - np.min(new_fea_map_show))
            new_fea_map_show = (new_fea_map_show - np.min(new_fea_map_show)) / np.max(new_fea_map_show)
        else:
            new_fea_map_show = (new_fea_map_show - np.min(new_fea_map_show)) / np.max(new_fea_map_show)

        new_fea_map_show = np.uint8(new_fea_map_show * 255.0)

        if self.inverse_set == True:
            new_fea_map_show = 255 - new_fea_map_show
        else:
            pass

        new_fea_map_show = cv2.applyColorMap(new_fea_map_show, cv2.COLORMAP_JET)
        new_weighted_image = cv2.addWeighted(vis_show_img, 0.8, new_fea_map_show, 0.4, 0.3)

        return (vis_show_img_gray, finding_img, stat_img, vis_show_img, xxxx, fea_map_show, weighted_image,
                new_fea_map_show, new_weighted_image, test_num, show_fea_np, pre_proba, pre_cls_num, vis_img_tr, ssss,
                finding_img_rgb)

    ########################## show_results #########################
    def show_results(self):
        (vis_show_img_gray, finding_img, stat_img, vis_show_img, xxxx, fea_map_show, weighted_image, new_fea_map_show,
         new_weighted_image, test_num, show_fea_np, pre_proba, pre_cls_num, vis_img_tr, ssss, _) = self.bese_function()

        plt.figure(1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(finding_img, cmap='gray')

        plt.figure(2)
        sns.histplot(stat_img, bins=5, color='black', stat='density', kde=True)

        plt.figure(3)
        plt.subplot(3, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(vis_show_img)

        plt.subplot(3, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(xxxx)

        plt.subplot(3, 2, 3)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(fea_map_show)

        plt.subplot(3, 2, 4)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(weighted_image)

        plt.subplot(3, 2, 5)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(new_fea_map_show)

        plt.subplot(3, 2, 6)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(new_weighted_image)

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
        else:
            plt.figure(4, figsize=(2, 1.5))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(show_fea_np[:, :, test_num])

        print('predict results:', pre_cls_num)
        print('predict probability:', pre_proba)

        if self.adaptive_strategy == 'adaptive':
            ######################### adaptive strategy #############################
            self.ads_cam_model.eval()
            with torch.no_grad():
                pre_cam_y = self.ads_cam_model(vis_img_tr.view(1, 3, 224, 224).cuda(self.gpu_device))

            pre_cam_cls_num = torch.argmax(nn.Softmax()(pre_cam_y)).detach().cpu().numpy()
            pre_cam_proba = nn.Softmax()(pre_cam_y).detach().cpu().numpy()
            pre_cam_proba = pre_cam_proba[0, pre_cam_cls_num]

            if pre_cam_cls_num == 1:
                final_cam = new_fea_map_show
                final_image = new_weighted_image
            elif pre_cam_cls_num == 0:
                final_cam = fea_map_show
                final_image = weighted_image
            else:
                print('error!!!')
        elif self.adaptive_strategy == 'histopathology':
            print(self.img_name)
            ##########################  strategy for histopathology images  ############################
            s1_sum_count = 0
            s2_sum_count = 0
            for i in range(ssss.shape[0]):
                for j in range(ssss.shape[1]):
                    if xxxx[i, j] != 0:
                        xxxx[i, j] = 1
                        s1_sum_count += 1

                    if ssss[i, j] != 0:
                        ssss[i, j] = 1
                        s2_sum_count += 1

            s_1 = xxxx * vis_show_img_gray
            s_2 = ssss * vis_show_img_gray
            s_1_count = 0
            s_2_count = 0

            s_1_max_count = 0
            s_2_max_count = 0

            for i in range(s_1.shape[0]):
                for j in range(s_1.shape[1]):
                    if s_1[i, j] < 0.6 and s_1[i, j] != 0:
                        s_1_count += 1

                    if s_2[i, j] < 0.6 and s_2[i, j] != 0:
                        s_2_count += 1

                    if s_1[i, j] > 0.85:
                        s_1_max_count += 1

                    if s_2[i, j] > 0.85:
                        s_2_max_count += 1

            if (s2_sum_count / (224 * 224)) < 0.08 or (s1_sum_count / (224 * 224)) < 0.08:
                if (s1_sum_count / (224 * 224)) < 0.08:
                    final_cam = new_fea_map_show
                    final_image = new_weighted_image
                elif (s2_sum_count / (224 * 224)) < 0.08:
                    final_cam = fea_map_show
                    final_image = weighted_image
            else:
                if (s_2_count / s2_sum_count) >= (s_1_count / s1_sum_count):
                    final_cam = new_fea_map_show
                    final_image = new_weighted_image
                elif (s_2_count / s2_sum_count) < (s_1_count / s1_sum_count):
                    final_cam = fea_map_show
                    final_image = weighted_image
                else:
                    print('error!!!')
        elif self.adaptive_strategy == 'natural':
            ##########################  strategy for natural images  ############################
            match_img_1 = \
            io.imread(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\ImageNet-Mini\Train\98\n0191074700000043.jpg')
            match_img_1_gray = color.rgb2gray(match_img_1)

            match_img_2 = \
            io.imread(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\ImageNet-Mini\Train\98\n0191074700000025.jpg')
            match_img_2_gray = color.rgb2gray(match_img_2)

            match_final_1 = transform.resize(match_img_1_gray, (224, 224))
            match_final_2 = transform.resize(match_img_2_gray, (224, 224))
            xxxx_img = xxxx * vis_show_img_gray
            ssss_img = ssss * vis_show_img_gray

            xxxx_values_1 = metrics.structural_similarity(match_final_1, xxxx_img, win_size=5)
            ssss_values_1 = metrics.structural_similarity(match_final_1, ssss_img, win_size=5)
            xxxx_values_2 = metrics.structural_similarity(match_final_2, xxxx_img, win_size=5)
            ssss_values_2 = metrics.structural_similarity(match_final_2, ssss_img, win_size=5)

            xxxx_count = 0
            ssss_count = 0
            for i in range(xxxx.shape[0]):
                for j in range(xxxx.shape[1]):
                    if xxxx[i, j] != 0:
                        xxxx_count += 1

                    if ssss[i, j] != 0:
                        ssss_count += 1

            xxxx_values = (xxxx_values_1 + xxxx_values_2) / (xxxx_count / (224 * 224))
            ssss_values = (ssss_values_1 + ssss_values_2) / (ssss_count / (224 * 224))

            if xxxx_values >= ssss_values:
                final_cam = fea_map_show
                final_image = weighted_image
            else:
                final_cam = fea_map_show
                final_image = weighted_image

            print(xxxx_values, ssss_values)
        else:
            print('error！！！')

        plt.figure(5)
        plt.subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(final_cam)

        plt.subplot(1, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(final_image)



        plt.show()
        io.imsave('E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\single\\new_114.jpg', new_weighted_image)


    ########################## get_training_images #########################
    def get_training_images(self):
        (vis_show_img_gray, finding_img, stat_img, vis_show_img, xxxx, fea_map_show, weighted_image, new_fea_map_show
         , new_weighted_image, test_num, show_fea_np, pre_proba, pre_cls_num, _, _,
         finding_img_rgb) = self.bese_function()

        plt.figure(1)
        plt.subplot(3, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(vis_show_img)

        plt.subplot(3, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(xxxx)

        plt.subplot(3, 2, 3)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(fea_map_show)

        plt.subplot(3, 2, 4)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(weighted_image)

        plt.subplot(3, 2, 5)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(new_fea_map_show)

        plt.subplot(3, 2, 6)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(new_weighted_image)

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
        else:
            pass

        print('predict results:', pre_cls_num)
        print('predict probability:', pre_proba)
        plt.show()

        training_cls = input('training_cls:')
        print(training_cls)

        if training_cls == str(0):
            io.imsave(self.training_img_path + '0\\' + str(pre_cls_num) + self.img_name, finding_img_rgb)
        elif training_cls == str(1):
            io.imsave(self.training_img_path + '1\\' + str(pre_cls_num) + self.img_name, finding_img_rgb)
        else:
            print('please input valid values!!!')

    ########################## get_all_training_images #########################
    def get_all_training_images(self):
        (vis_show_img_gray, finding_img, stat_img, vis_show_img, xxxx, fea_map_show, weighted_image, new_fea_map_show
         , new_weighted_image, test_num, show_fea_np, pre_proba, pre_cls_num, _, _,
         finding_img_rgb) = self.bese_function()

        io.imsave(self.training_img_path + self.img_name[:-4] + '_0' + self.img_name[-4:], weighted_image)
        io.imsave(self.training_img_path + self.img_name[:-4] + '_1' + self.img_name[-4:], new_weighted_image)
        print(self.img_name)

    ########################## get_result_images #########################
    def get_result_images(self):
        (vis_show_img_gray, finding_img, stat_img, vis_show_img, xxxx, fea_map_show, weighted_image,
        new_fea_map_show, new_weighted_image, test_num, show_fea_np, pre_proba, pre_cls_num, vis_img_tr, ssss,
        finding_img_rgb) = self.bese_function()

        if self.adaptive_strategy == 'adaptive':
            ######################### adaptive strategy #############################
            self.ads_cam_model.eval()
            with torch.no_grad():
                pre_cam_y = self.ads_cam_model(vis_img_tr.view(1, 3, 224, 224).cuda(self.gpu_device))

            pre_cam_cls_num = torch.argmax(nn.Softmax()(pre_cam_y)).detach().cpu().numpy()
            pre_cam_proba = nn.Softmax()(pre_cam_y).detach().cpu().numpy()
            pre_cam_proba = pre_cam_proba[0, pre_cam_cls_num]

            if pre_cam_cls_num == 1:
                final_cam = new_fea_map_show
                final_image = new_weighted_image
            elif pre_cam_cls_num == 0:
                final_cam = fea_map_show
                final_image = weighted_image
            else:
                print('error!!!')
        elif self.adaptive_strategy == 'histopathology':
            ##########################  strategy for histopathology images  ############################
            s1_sum_count = 0
            s2_sum_count = 0
            for i in range(ssss.shape[0]):
                for j in range(ssss.shape[1]):
                    if xxxx[i, j] != 0:
                        xxxx[i, j] = 1
                        s1_sum_count += 1

                    if ssss[i, j] != 0:
                        ssss[i, j] = 1
                        s2_sum_count += 1

            s_1 = xxxx * vis_show_img_gray
            s_2 = ssss * vis_show_img_gray
            s_1_count = 0
            s_2_count = 0

            s_1_max_count = 0
            s_2_max_count = 0

            for i in range(s_1.shape[0]):
                for j in range(s_1.shape[1]):
                    if s_1[i, j] < 0.5 and s_1[i, j] != 0:
                        s_1_count += 1

                    if s_2[i, j] < 0.5 and s_2[i, j] != 0:
                        s_2_count += 1

                    if s_1[i, j] > 0.8:
                        s_1_max_count += 1

                    if s_2[i, j] > 0.8:
                        s_2_max_count += 1

            if (s2_sum_count / (224 * 224)) < 0.08 or (s1_sum_count / (224 * 224)) < 0.08:
                if (s1_sum_count / (224 * 224)) < 0.08:
                    final_cam = new_fea_map_show
                    final_image = new_weighted_image
                elif (s2_sum_count / (224 * 224)) < 0.08:
                    final_cam = fea_map_show
                    final_image = weighted_image
            else:
                if (s_2_count / s2_sum_count) >= (s_1_count / s1_sum_count):
                    final_cam = new_fea_map_show
                    final_image = new_weighted_image
                elif (s_2_count / s2_sum_count) < (s_1_count / s1_sum_count):
                    final_cam = fea_map_show
                    final_image = weighted_image
                else:
                    print('error!!!')
        elif self.adaptive_strategy == 'natural':
            ##########################  strategy for natural images  ############################
            match_img_1 = \
                io.imread(
                    r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\ImageNet-Mini\Train\98\n0191074700000043.jpg')
            match_img_1_gray = color.rgb2gray(match_img_1)

            match_img_2 = \
                io.imread(
                    r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\ImageNet-Mini\Train\98\n0191074700000025.jpg')
            match_img_2_gray = color.rgb2gray(match_img_2)

            match_final_1 = transform.resize(match_img_1_gray, (224, 224))
            match_final_2 = transform.resize(match_img_2_gray, (224, 224))
            xxxx_img = xxxx * vis_show_img_gray
            ssss_img = ssss * vis_show_img_gray

            xxxx_values_1 = metrics.structural_similarity(match_final_1, xxxx_img, win_size=5)
            ssss_values_1 = metrics.structural_similarity(match_final_1, ssss_img, win_size=5)
            xxxx_values_2 = metrics.structural_similarity(match_final_2, xxxx_img, win_size=5)
            ssss_values_2 = metrics.structural_similarity(match_final_2, ssss_img, win_size=5)

            xxxx_count = 0
            ssss_count = 0
            for i in range(xxxx.shape[0]):
                for j in range(xxxx.shape[1]):
                    if xxxx[i, j] != 0:
                        xxxx_count += 1

                    if ssss[i, j] != 0:
                        ssss_count += 1

            xxxx_values = (xxxx_values_1 + xxxx_values_2) / (xxxx_count / (224 * 224))
            ssss_values = (ssss_values_1 + ssss_values_2) / (ssss_count / (224 * 224))

            if xxxx_values >= ssss_values:
                final_cam = fea_map_show
                final_image = weighted_image
            else:
                final_cam = fea_map_show
                final_image = weighted_image

            print(xxxx_values, ssss_values)
        else:
            print('error！！！')

        io.imsave(self.result_img_path + r'\\' + self.img_name, final_image)
        io.imsave(self.result_cam_path + r'\\' + self.img_name, final_cam)
        print(self.img_name)

    ########################## get_result_images #########################
    def get_all_result_images(self):
        (vis_show_img_gray, finding_img, stat_img, vis_show_img, xxxx, fea_map_show, weighted_image,
            new_fea_map_show, new_weighted_image, test_num, show_fea_np, pre_proba, pre_cls_num, vis_img_tr,
            ssss, finding_img_rgb) = self.bese_function()

        io.imsave(self.result_img_path + r'\\' + self.img_name[:-4] + r'_0.jpg', weighted_image)
        io.imsave(self.result_img_path + r'\\' + self.img_name[:-4] + r'_1.jpg', new_weighted_image)
        io.imsave(self.result_cam_path + r'\\' + self.img_name[:-4] + r'_0.jpg', fea_map_show)
        io.imsave(self.result_cam_path + r'\\' + self.img_name[:-4] + r'_1.jpg', new_fea_map_show)
        print(self.img_name)




