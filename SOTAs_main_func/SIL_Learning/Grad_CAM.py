############################# Grad-CAM ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing Grad-CAM

########################## API Section #########################
import skimage.color
import torch
from torch import nn
from torch.autograd import Variable
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


class Grad_CAM:
    def __init__(self, model = None, path_name = None,  inverse_set = True,
                 transform = None, gpu_device = 0, show_all_fp = True, map_size = [7, 7], model_name = 'swint',
                 training_img_path = None, img_name = None, result_img_path = None, result_cam_path = None,
                 out_mode = None):
        self.model = model
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
        self.map_size = map_size
        self.model_name = model_name
        self.out_mode = out_mode

    def bese_function(self):
        finding_img_rgb = io.imread(self.path_name + r'/' + self.img_name)
        finding_img = skimage.color.rgb2gray(finding_img_rgb)
        stat_img = finding_img.reshape(finding_img.shape[0] * finding_img.shape[1])

        show_fea = dict()

        show_grad = dict()

        def hook_fn(model, in_fea, out_fea):
            show_fea['out_fea'] = out_fea
            print("hook_fn 被调用了")
            print("show_grad 中的键:", show_fea.keys())

        def hook_grad_fn(model, in_grad, out_grad):
            show_grad['out_grad'] = out_grad
            print("hook_grad_fn 被调用了")
            print("show_grad 中的键:", show_grad.keys())


        transform_img = transforms.Resize([224, 224], interpolation=InterpolationMode.NEAREST)
        vis_img = PIL.Image.open(self.path_name + r'/' + self.img_name)
        vis_show_img = transform_img(vis_img)
        vis_show_img = np.array(vis_show_img)
        vis_show_img_gray = skimage.color.rgb2gray(vis_show_img)

        vis_img_tr = self.transform(vis_img)
        print(self.out_mode)
        ##########
        self.model.eval()
        if self.model_name == 'Swin_T' or self.model_name == 'Mine':
            self.model.norm.register_forward_hook(hook_fn)
            self.model.norm.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'ResNext':
            self.model.base.layer4[1].conv3.register_forward_hook(hook_fn)
            self.model.base.layer4[1].conv3.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'ResNet50':
            self.model.layer4[1].conv3.register_forward_hook(hook_fn)
            self.model.layer4[1].conv3.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'DenseNet121':
            self.model.base_net.norm5.register_forward_hook(hook_fn)
            self.model.base_net.norm5.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'FABNeT':
            self.model.BN_5.register_forward_hook(hook_fn)
            self.model.BN_5.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'T2T_ViT':
            self.model.base.norm.register_forward_hook(hook_fn)
            self.model.base.norm.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'MViT':
            self.model.base.norm3.register_forward_hook(hook_fn)
            self.model.base.norm3.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'ViG':
            self.model.block_11.register_forward_hook(hook_fn)
            self.model.block_11.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'ViT_AMC':
            self.model.transformer_block_11.mlp.fc2.register_forward_hook(hook_fn)
            self.model.transformer_block_11.mlp.fc2.register_backward_hook(hook_grad_fn)


        elif self.model_name == 'twins':
            self.model.base.norm_after_stage3.register_forward_hook(hook_fn)
            self.model.base.norm_after_stage3.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'poolformer':
            self.model.base.norm6.register_forward_hook(hook_fn)
            self.model.base.norm6.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'efficient':
            self.model.base.layers[6].conv.register_forward_hook(hook_fn)
            self.model.base.layers[6].conv.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'convnext':
            self.model.base.stages[3][2].drop_path.register_forward_hook(hook_fn)
            self.model.base.stages[3][2].drop_path.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'repvgg':
            self.model.base.stage_4[0].act.register_forward_hook(hook_fn)
            self.model.base.stage_4[0].act.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'mlp':
            self.model.base.ln1.register_forward_hook(hook_fn)
            self.model.base.ln1.register_backward_hook(hook_grad_fn)
        elif self.model_name == 'vit':
            self.model.transformer_block_11.mlp.fc2.register_forward_hook(hook_fn)
            self.model.transformer_block_11.mlp.fc2.register_backward_hook(hook_grad_fn)

        if self.out_mode == 'single':
            pre_y = self.model(vis_img_tr.view(1, 3, 224, 224).cuda(self.gpu_device))
        elif self.out_mode == 'triplet':
            _, _, pre_y = self.model(vis_img_tr.view(1, 3, 224, 224).cuda(self.gpu_device))
        elif self.out_mode == 'five':
            _, _, pre_y,_,_ = self.model(vis_img_tr.view(1, 3, 224, 224).cuda(self.gpu_device))
        cls_idx = torch.argmax(pre_y).item()  # 获取预测类别编码
        score = pre_y[:, cls_idx].sum()  # 获取预测类别分数
        #self.model.zero_grad()
        score.backward(retain_graph=True)  # 由预测类别分数反向传播

        pre_cls_num = torch.argmax(nn.Softmax()(pre_y)).detach().cpu().numpy()
        pre_proba = nn.Softmax()(pre_y).detach().cpu().numpy()
        pre_proba = pre_proba[0, pre_cls_num]

        show_fea_np = show_fea['out_fea'].detach().cpu()
        avgp_weight = show_grad['out_grad'][0]

        if self.model_name == 'ResNext':
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[1])
            avgp_weight = torch.mean(avgp_weight, dim=[2, 3], keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[1], ))
        elif self.model_name == 'ViG':
            poll = nn.AvgPool2d(kernel_size=2, stride=2)
            show_fea_np = poll(show_fea_np)
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1],show_fea_np.shape[1])
            avgp_weight = torch.mean(avgp_weight, dim=[2, 3], keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[1],))
        elif self.model_name == 'Swin_T' or self.model_name == 'Mine':
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[2])
            avgp_weight = torch.mean(avgp_weight, dim=1, keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[2],))
        elif self.model_name == 'ResNet50':
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[1])
            avgp_weight = torch.mean(avgp_weight, dim=[2, 3], keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[1],))
        elif self.model_name == 'DenseNet121' or self.model_name == 'FABNeT':
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[1])
            avgp_weight = torch.mean(avgp_weight, dim=[2, 3], keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[1],))
        elif self.model_name == 'ViT_AMC':
            show_fea_np = show_fea_np[:, 1:, :]
            # show_fea_np = show_fea_np.reshape(1, show_fea_np.shape[2], 14, 14)
            # poll = nn.AvgPool2d(kernel_size=2, stride=2)
            # show_fea_np = poll(show_fea_np)
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[2])
            avgp_weight = avgp_weight[:,1:, :]
            avgp_weight = torch.mean(avgp_weight, dim=1, keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[2],))
        elif self.model_name == 'vit' or self.model_name == 'T2T_ViT':
            show_fea_np = show_fea_np[:, 1:, :]
            # show_fea_np = show_fea_np.reshape(1, show_fea_np.shape[2], 14, 14)
            # poll = nn.AvgPool2d(kernel_size=2,stride=2)
            # show_fea_np = poll(show_fea_np)
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[2])
            avgp_weight = avgp_weight[:,1:, :]
            avgp_weight = torch.mean(avgp_weight, dim=1, keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[2],))
        elif self.model_name == 'MViT':
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[2])
            avgp_weight = torch.mean(avgp_weight, dim=1, keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[2],))

        elif self.model_name == 'mlp':
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[2])
            avgp_weight = torch.mean(avgp_weight, dim=1, keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[2],))
        elif self.model_name == 'twins':
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[2])
            avgp_weight = torch.mean(avgp_weight, dim=1, keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[2],))
        else:
            show_fea_np = show_fea_np.reshape(self.map_size[0], self.map_size[1], show_fea_np.shape[1])
            avgp_weight = torch.mean(avgp_weight, dim=[2, 3], keepdim=True)
            avgp_weight = torch.reshape(avgp_weight, (avgp_weight.shape[1],))
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.model_name == 'vit' or self.model_name == 'ViT_AMC':
                    show_fea_np[i, j, :] = show_fea_np[i, j, :]
                else:
                    show_fea_np[i, j, :] = show_fea_np[i, j, :] * avgp_weight.detach().cpu()
        fea_map = show_fea_np * 1

        fea_map_show = torch.mean(fea_map, dim=2)
        fea_map_show = (fea_map_show - torch.mean(fea_map_show)) / torch.std(fea_map_show)
        fea_map_show = fea_map_show - torch.min(fea_map_show)

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if fea_map_show[i, j] <=0:
                    fea_map_show[i, j] = 0

        if self.model_name == 'ResNext':
            rd_rate = 0.7
        elif self.model_name == 'MViT' or self.model_name == 'T2T_ViT':
            rd_rate = 0.95
        elif self.model_name == 'repvgg' or self.model_name == 'poolformer' or self.model_name == 'convnext':
            rd_rate = 0.7
        elif self.model_name == 'mlp' or self.model_name == 'Swin_T':
            rd_rate = 0.9
        elif self.model_name == 'twins':
            rd_rate = 0.97
        else:
            rd_rate = 0.85


        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if fea_map_show[i, j] <= rd_rate * torch.max(fea_map_show):
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

        io.imsave(self.result_img_path + r'/' + self.img_name, weighted_image)
        #io.imsave(self.result_cam_path + r'/' + self.img_name, fea_map_show)
        print(self.img_name)