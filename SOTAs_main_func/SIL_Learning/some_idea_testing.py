import numpy as np
import torch
from torch import nn
import warnings
import os
import pandas as pd
import natsort
from skimage import io
warnings.filterwarnings('ignore')

inter_grad = []
def grad_fn(grad):
    inter_grad.append(grad)
    return 2 * grad

if __name__ == '__main__':

    '''
    a = torch.Tensor([1, 2]).requires_grad_()
    b = torch.Tensor([3, 4]).requires_grad_()
    d = torch.Tensor([2]).requires_grad_()
    c = a + b
    e = c * d
    o = e.sum()
    c.register_hook(grad_fn)
    o.backward()
    print(inter_grad)

    print(a.grad)
    print(b.grad)
    print(d.grad)

    print(c.grad)
    print(e.grad)
    print(o.grad)
    '''

    #for i in range(100):
    #    os.makedirs(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\mini-imagenet\Val\\'+str(i))

    '''
    excel_name = 'train.csv'
    train_csv_data = pd.read_csv(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\mini-imagenet\\' + excel_name)
    train_np_data = train_csv_data.values
    train_np_label = train_np_data[:, 1]
    label_name = np.unique(train_np_label)
    if excel_name == 'train.csv':
        fir_name = np.arange(0, label_name.shape[0])
    elif excel_name == 'test.csv':
        fir_name = np.arange(0, label_name.shape[0]) + 64
    elif excel_name == 'val.csv':
        fir_name = np.arange(0, label_name.shape[0]) + 64 + 20
    target_name = np.concatenate(
        (fir_name.reshape(fir_name.shape[0], 1), label_name.reshape(label_name.shape[0], 1)), axis=1)
    print(train_np_data[0, 0])
    print(train_np_data[0, 1])

    read_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\mini-imagenet\images\\'
    save_train_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\mini-imagenet\Train\\'
    save_test_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\mini-imagenet\Test\\'
    save_val_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\mini-imagenet\Val\\'
    save_fir = 0
    fir_class = 0

    for i in range(train_np_data.shape[0]):
        fir_class = i % 600
        img = io.imread(read_path + train_np_data[i, 0])
        for j in range(target_name.shape[0]):
            if train_np_data[i, 1] == target_name[j, 1]:
                save_fir = target_name[j, 0]

        if fir_class < 480:
            io.imsave(save_train_path + str(save_fir) + '\\' + train_np_data[i, 0], img)
        elif fir_class < 540:
            io.imsave(save_val_path + str(save_fir) + '\\' + train_np_data[i, 0], img)
        else:
            io.imsave(save_test_path + str(save_fir) + '\\' + train_np_data[i, 0], img)
        print('success read:' + str(i))

    '''

    a = input('input:')
    print(a)


