import torch
from torch import nn
import time
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from torch import optim
import pandas as pd
import random
import warnings
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')

########################## learning functions #########################
def cnn_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-4
    elif epoch < 75:
        lr = 2e-5
    else:
        lr = 1e-6
    return lr

def vit_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-5
    elif epoch < 75:
        lr = 5e-6
    else:
        lr = 1e-6
    return lr

def vit_lr_for_breast_schedule(epoch):
    if epoch < 50:
        lr = 6e-6
    elif epoch < 75:
        lr = 1e-6
    else:
        lr = 1e-7
    return lr

def fusion_lr_schedule(epoch):
    if epoch < 50:
        lr = 5e-5
    elif epoch < 75:
        lr = 2e-6
    else:
        lr = 1e-6
    return lr

def one_hot(org_x = None, pre_dim = 3):
    one_x = np.zeros((org_x.shape[0], pre_dim))
    for i in range(org_x.shape[0]):
        one_x[i, int(org_x[i])] = 1
    return one_x


########################## testing function #########################
def testing_funnction(test_model = None, train_loader=None, val_loader=None, test_loader=None, gpu_device=1,
                      out_mode = None):
    loss_fn = nn.CrossEntropyLoss()
    test_model.eval()
    train_acc = []
    for train_img, train_label in train_loader:
        train_img = train_img.cuda(gpu_device)
        train_label = train_label.cuda(gpu_device)
        with torch.no_grad():
            if out_mode == 'single':
                train_pre_y = test_model(train_img)
            elif out_mode == 'triplet':
                _, _, train_pre_y = test_model(train_img)
            train_pre_label = torch.argmax(train_pre_y, dim=1)
            train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                            train_pre_label.detach().cpu().numpy()))

    val_acc = []
    val_all_label = np.zeros(1)
    val_all_pre_label = np.zeros(1)
    for val_img, val_label in val_loader:
        val_img = val_img.cuda(gpu_device)
        val_label = val_label.cuda(gpu_device)
        with torch.no_grad():
            if out_mode == 'single':
                val_pre_y = test_model(val_img)
            elif out_mode == 'triplet':
                _, _, val_pre_y = test_model(val_img)
            val_loss = loss_fn(val_pre_y, val_label)
            val_pre_label = torch.argmax(val_pre_y, dim=1)
            val_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                          val_pre_label.detach().cpu().numpy()))
            val_all_label = np.concatenate((val_all_label, val_label.detach().cpu().numpy()))
            val_all_label = val_all_label[1:]
            val_all_pre_label = np.concatenate((val_all_pre_label, val_pre_label.detach().cpu().numpy()))
            val_all_pre_label = val_all_pre_label[1:]

    test_model.eval()
    test_acc = []
    test_all_label = np.zeros(1)
    test_all_pre_label = np.zeros(1)
    test_all_pre_proba = np.zeros((1, 3))
    for test_img, test_label in test_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            if out_mode == 'single':
                test_pre_y = test_model(test_img)
            elif out_mode == 'triplet':
                _, _, test_pre_y = test_model(test_img)

            test_loss = loss_fn(test_pre_y, test_label)
            test_pre_label = torch.argmax(test_pre_y, dim=1)
            test_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                        test_pre_label.detach().cpu().numpy()))
            test_all_label = np.concatenate((test_all_label, test_label.detach().cpu().numpy()))
            test_all_label = test_all_label[1:]
            test_all_pre_label = np.concatenate((test_all_pre_label, test_pre_label.detach().cpu().numpy()))
            test_all_pre_proba = np.concatenate(
                (test_all_pre_proba, torch.softmax(test_pre_y, dim=1).detach().cpu().numpy()))
            test_all_pre_label = test_all_pre_label[1:]
    test_all_proba = one_hot(org_x = test_all_label, pre_dim = 3)
    test_all_pre_proba = one_hot(org_x= test_all_pre_label, pre_dim = 3)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
        ' val_acc:{:.4}'.format(np.mean(val_acc)),
        ' test_acc:{:.4}'.format(np.mean(test_acc)), '\n')

    print('########################## validation set results #########################')
    print(classification_report(val_all_label, val_all_pre_label, digits=4))

    print('########################## testing set results #########################')
    print(classification_report(test_all_label, test_all_pre_label, digits=4))

    print('########################## auc results #########################')
    print(roc_auc_score(test_all_proba, test_all_pre_proba))


def Single_out_fit(net=None, train_loader=None, val_loader=None, test_loader=None, epoch=100,
                   gpu_device=1, weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1_.pth'):
    loss_fn = nn.CrossEntropyLoss()
    print('########################## training results #########################')
    for i in range(epoch):
        start_time = time.time()
        optim = torch.optim.SGD(net.parameters(), lr=0.001)
        net = net.cuda(gpu_device)

        net.train()
        for img_data, img_label in train_loader:
            img_data = img_data.cuda(gpu_device)
            img_label = img_label.cuda(gpu_device)
            pre_y = net(img_data)
            loss_value = loss_fn(pre_y, img_label)
            loss_value.backward()
            optim.step()
            optim.zero_grad()

        net.eval()
        train_acc = []
        for train_img, train_label in train_loader:
            train_img = train_img.cuda(gpu_device)
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                train_pre_y = net(train_img)
                train_pre_label = torch.argmax(train_pre_y, dim=1)
                train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                            train_pre_label.detach().cpu().numpy()))

        val_acc = []
        for val_img, val_label in val_loader:
            val_img = val_img.cuda(gpu_device)
            val_label = val_label.cuda(gpu_device)
            with torch.no_grad():
                val_pre_y = net(val_img)
                val_loss = loss_fn(val_pre_y, val_label)
                val_pre_label = torch.argmax(val_pre_y, dim=1)
                val_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                            val_pre_label.detach().cpu().numpy()))

        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(loss_value.detach().cpu().numpy()),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(val_loss.detach().cpu().numpy()),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    net.eval()
    test_acc = []
    for test_img, test_label in test_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            test_pre_y = net(test_img)
            test_loss = loss_fn(test_pre_y, test_label)
            test_pre_label = torch.argmax(test_pre_y, dim=1)
            test_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               test_pre_label.detach().cpu().numpy()))


    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))

    g = net.state_dict()
    torch.save(g, weight_path)

    return test_acc






