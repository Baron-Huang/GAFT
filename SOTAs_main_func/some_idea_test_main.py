from skimage import io, color, transform
import os
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import random


if __name__ == '__main__':

    '''
    target_cam_name = os.listdir(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\ADS_CAM_dataset\Brain_cancer\1')
    search_img_name = os.listdir(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Brain_dataset\Train_patch\IV')
    save_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\ADS_CAM_dataset\\New_Brain_cancer\1'
    search_path_name = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Brain_dataset\Train_patch\IV'

    for i in target_cam_name:
        for j in search_img_name:
            if i[0] == '3' and i[1:] == j:
                transfer_img = io.imread(search_path_name + '\\' + j)
                io.imsave(save_path + '\\' + i, transfer_img)
                print(j)
    print()
    '''

    '''
    mask_1 = io.imread(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\SSIM\n0153282900001175\label.png')
    mask_2 = io.imread(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\SSIM\n0153282900001183\label.png')

    mask_1_gray = color.rgb2gray(mask_1)
    mask_2_gray = color.rgb2gray(mask_2)

    for i in range(mask_1_gray.shape[0]):
        for j in range(mask_1_gray.shape[1]):
            if mask_1_gray[i, j] != 0:
                mask_1_gray[i, j] = 1
            else:
                mask_1_gray[i, j] = 0

    for i in range(mask_2_gray.shape[0]):
        for j in range(mask_2_gray.shape[1]):
            if mask_2_gray[i, j] != 0:
                mask_2_gray[i, j] = 1
            else:
                mask_2_gray[i, j] = 0

    image_1 = io.imread(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\ImageNet-Mini\Test\00\\n0153282900001175.jpg')
    image_2 = io.imread(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\ImageNet-Mini\Test\00\\n0153282900001183.jpg')

    image_1_gray = color.rgb2gray(image_1)
    image_2_gray = color.rgb2gray(image_2)
    #image_1_gray = image_1_gray * mask_1_gray
    #image_2_gray = image_2_gray * mask_2_gray

    image_1_gray = transform.resize(image_1_gray, (224, 224))
    image_2_gray = transform.resize(image_2_gray, (224, 224))

    for i in range(image_1_gray.shape[0]):
        for j in range(image_1_gray.shape[1]):
            if image_1_gray[i, j] == 0:
                image_1_gray[i, j] = np.random.rand(1)

            #if image_2_gray[i, j] == 0:
                #image_2_gray[i, j] = (np.random.rand(1) + np.random.rand(1)) / 2

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_1_gray, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mask_1_gray)

    plt.subplot(2, 2, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image_2_gray, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mask_2_gray)

    ssim_values = structural_similarity(image_1_gray, image_2_gray, multichannel=False, win_size=3)
    print(ssim_values)

    plt.show()
    '''

    '''
    search_path = os.listdir(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\\Natural_images\2')
    img_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\ImageNet-Mini\Train\02'
    save_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\\Natural_images\mixed'

    for i in search_path:
        print(i[-5])
        #print(img_path + '\\' + i[:-6] + i[-4:])
        img_get = io.imread(img_path + '\\' + i[:-6] + i[-4:])
        io.imsave(save_path + '\\' + i[-5] + '\\' + i[:-6] + i[-4:], img_get)

    pass  
    '''

    '''
    xml_name = os.listdir(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\Annotations')

    for i in xml_name:
        dom = ET.parse(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\Annotations\\' + i)
        cls_name = dom.find('object').find('name').text
        img_name = dom.find('filename').text
        name_encoder = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6,'cat': 7,
                        'chair': 8,'cow': 9,'diningtable': 10, 'dog': 11,'horse': 12,'motorbike': 13, 'person': 14,
                        'pottedplant': 15,'sheep': 16,'sofa': 17,'train': 18,'tvmonitor': 19}
        img_get = io.imread(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\JPEGImages\\' + img_name)
        io.imsave(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\cls_datasets\\' +
                  str(name_encoder[cls_name]) + '\\' + img_name, img_get)

        print(name_encoder[cls_name])
        print(img_name)
    '''

    '''
    name_encoder = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
                    'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                    'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    cub_name = os.listdir(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CUB_200_2011\images')
    voc_name = ['00.aeroplane','01.bicycle', '02.bird', '03.boat', '04.bottle', '05.bus', '06.car', '07.cat',
                '08.chair', '08.cow', '10.diningtable', '11.dog', '12.horse', '13.motorbike', '14.person',
                '15.pottedplant', '16.sheep', '17.sofa', '18.train', '19.tvmonitor']

    voc_org_names = os.listdir(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\Original_imgs')
    '''

    #for i in range(200):
    #    os.makedirs(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CUB_200_2011\Train\\' + cub_name[i])
    #    #os.makedirs(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\cls_datasets\Val\\' + voc_name[i])
    #    os.makedirs(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CUB_200_2011\Test\\' + cub_name[i])

    '''
    for i in range(len(voc_org_names)):
        img_name = os.listdir(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\Original_imgs\\'
                              + voc_org_names[i])
        print(len(img_name))
        for j in range(len(img_name)):
            img_inter = io.imread(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\Original_imgs\\'
                                  + voc_org_names[i] + '\\' + img_name[j])
            if j < len(img_name) * 0.7:
                io.imsave(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\cls_datasets\Train\\'
                          + voc_name[i] + '\\' + img_name[j], img_inter)
            elif j < len(img_name) * 0.85:
                io.imsave(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\cls_datasets\Val\\'
                          + voc_name[i] + '\\' + img_name[j], img_inter)
            else:
                io.imsave(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\CLS_VOC2012\cls_datasets\Test\\'
                          + voc_name[i] + '\\' + img_name[j], img_inter)
    '''

    '''
    img_file_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_datasets_ST_MSLNet_Random\III'
    train_file_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_datasets_ST_MSLNet_Random\Train\III'
    test_file_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_datasets_ST_MSLNet_Random\Test\III'
    val_file_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_datasets_ST_MSLNet_Random\Val\III'
    img_name = os.listdir(img_file_path)
    b = np.arange(len(img_name))
    sampling_name = np.random.choice(b, size=len(img_name), replace=False)
    test_sam = np.unique(sampling_name)


    for i in range(len(img_name)):
        img_inter = io.imread(img_file_path + '\\' + img_name[sampling_name[i]])
        if i <= len(img_name) * 0.7:
            io.imsave(train_file_path + '\\' + img_name[sampling_name[i]], img_inter)
        elif i <= len(img_name) * 0.85:
            io.imsave(val_file_path + '\\' + img_name[sampling_name[i]], img_inter)
        else:
            io.imsave(test_file_path + '\\' + img_name[sampling_name[i]], img_inter)
        print(i, img_name[sampling_name[i]])
    '''
    read_dir = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_china\Val\III'
    save_dir = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\test_new_datasets\Val\III'
    image_names = os.listdir(read_dir)

    for i in image_names:
        inter_img = io.imread(read_dir + '\\' + i)
        inter_img = transform.resize(inter_img, (224, 224))
        io.imsave(save_dir + '\\' + i, inter_img)
        print(i)




    


