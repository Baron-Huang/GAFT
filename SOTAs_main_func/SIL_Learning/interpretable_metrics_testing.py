############################# Interpretable_Metrics ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing Interpretable_Metrics

########################## API Section #########################
import os
from skimage import io
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

def iou_ss_show(cam_dir = None):
    for test_num in range(len(cam_dir_1)):
        cam_example = io.imread(cam_path_1 + r'\\' + cam_dir_1[test_num])
        mask_example = io.imread(mask_path_1 + r'\\' + mask_dir_1[test_num])
        for j in range(224):
            for k in range(224):
                if np.sum(cam_example[j, k, :]) < 255:
                    cam_example[j, k, :] = np.zeros_like(cam_example[j, k, :])

        cam_example_gray = rgb2gray(cam_example)
        mask_example_gray = rgb2gray(mask_example)
        mask_example_gray = resize(mask_example_gray, (224, 224))

        for j in range(224):
            for k in range(224):
                if mask_example_gray[j, k] != 0:
                    mask_example_gray[j, k] = 1

                if cam_example_gray[j, k] != 0:
                    cam_example_gray[j, k] = 1

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(mask_example_gray)

        plt.subplot(1, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(cam_example_gray)

        A_or_B = np.sum(mask_example_gray) + np.sum(cam_example_gray)
        A_and_B = np.sum(mask_example_gray * cam_example_gray)

        print(A_or_B)
        print(A_and_B)
        print(A_and_B / A_or_B)

        ss_value = structural_similarity(mask_example_gray, cam_example_gray, win_size=5)
        print(ss_value)

        plt.show()

def iou_ss_compute(cam_dir = None, cam_path = None, mask_path = None):
    iou_sum = []
    ss_sum = []
    for test_num in range(len(cam_dir)):
        print(test_num)
        cam_example = io.imread(cam_path_1 + r'\\' + cam_dir_1[test_num])
        mask_example = io.imread(mask_path_1 + r'\\' + mask_dir_1[test_num])
        for j in range(224):
            for k in range(224):
                if np.sum(cam_example[j, k, :]) < 255:
                    cam_example[j, k, :] = np.zeros_like(cam_example[j, k, :])

        cam_example_gray = rgb2gray(cam_example)
        mask_example_gray = rgb2gray(mask_example)
        mask_example_gray = resize(mask_example_gray, (224, 224))

        for j in range(224):
            for k in range(224):
                if mask_example_gray[j, k] != 0:
                    mask_example_gray[j, k] = 1

                if cam_example_gray[j, k] != 0:
                    cam_example_gray[j, k] = 1

        A_or_B = np.sum(mask_example_gray) + np.sum(cam_example_gray)
        A_and_B = np.sum(mask_example_gray * cam_example_gray)

        #print(A_or_B)
        #print(A_and_B)
        #print(A_and_B / A_or_B)

        ss_value = structural_similarity(mask_example_gray, cam_example_gray, win_size=5)
        #print(ss_value)
        iou_sum.append(A_and_B / A_or_B)
        ss_sum.append(ss_value)
    return iou_sum, ss_sum


if __name__ == '__main__':
    mask_sim_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_china\Masks'
    cam_sim_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_china\SwinT_Smooth_Grad_CAM_PP\CAMs'

    mask_path_1 = mask_sim_path + r'\\' + r'I'
    cam_path_1 =  cam_sim_path + r'\\' + r'I'
    mask_dir_1 = os.listdir(mask_path_1)
    cam_dir_1 = os.listdir(cam_path_1)

    mask_path_2 = mask_sim_path + r'\\' + r'II'
    cam_path_2 = cam_sim_path + r'\\' + r'II'
    mask_dir_2 = os.listdir(mask_path_2)
    cam_dir_2 = os.listdir(cam_path_2)

    mask_path_3 = mask_sim_path + r'\\' + r'III'
    cam_path_3 = cam_sim_path + r'\\' + r'III'
    mask_dir_3 = os.listdir(mask_path_3)
    cam_dir_3 = os.listdir(cam_path_3)

    iou_values_1, ss_values_1 = iou_ss_compute(cam_dir = cam_dir_1, mask_path= mask_path_1, cam_path= cam_path_1)
    iou_values_2, ss_values_2 = iou_ss_compute(cam_dir = cam_dir_2, mask_path= mask_path_2, cam_path= cam_path_2)
    iou_values_3, ss_values_3 = iou_ss_compute(cam_dir = cam_dir_3, mask_path= mask_path_3, cam_path= cam_path_3)

    print('class_1_iou:',np.mean(iou_values_1))
    print('class_1_ss:', np.mean(ss_values_1))

    print('class_2_iou:', np.mean(iou_values_2))
    print('class_2_ss:', np.mean(ss_values_2))

    print('class_3_iou:', np.mean(iou_values_3))
    print('class_3_ss:', np.mean(ss_values_3))


    print('iou:', (np.mean(iou_values_1) * len(cam_dir_1) + np.mean(iou_values_2) * len(cam_dir_2) +
           np.mean(iou_values_3) * len(cam_dir_3)) / (len(cam_dir_1) + len(cam_dir_2) + len(cam_dir_3)))

    print('ss:', (np.mean(ss_values_1) * len(cam_dir_1) + np.mean(ss_values_2) * len(cam_dir_2) +
           np.mean(ss_values_3) * len(cam_dir_3)) / (len(cam_dir_1) + len(cam_dir_2) + len(cam_dir_3)))








