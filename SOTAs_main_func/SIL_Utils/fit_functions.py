############################# fit_functions ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: fitting functions for ViT models

########################## API Section #########################
import torch
from torch import nn
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from torch import optim
import pandas as pd
from SOTAs_main_func.SIL_Model.ViT_models.ViT import VisionTransformer
from SOTAs_main_func.SIL_Model.ViT_models.ViT_model_modules import ViT_Net
import random
from SOTAs_main_func.SIL_Model.SwinT_models.models.SwinT_model_modules import creating_swinT
from SOTAs_main_func.SIL_Model.ViT_models.ViT_model_modules import creating_ViT
from SOTAs_main_func.SIL_Model.Mixer_models.models.Mixer_model_modules import creating_Mixer
import warnings
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
warnings.filterwarnings('ignore')

########################## seed_function #########################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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


def confusion_matrix_metrics(y_true, y_pred, labels):
    """Build a confusion matrix and per-class metrics derived from it."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(matrix.sum())
    rows = [
        'Class  TP  FP  FN  TN  Sensitivity  Specificity  Precision  F1-score'
    ]
    for class_index, label in enumerate(labels):
        true_positive = int(matrix[class_index, class_index])
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        true_negative = total - true_positive - false_positive - false_negative
        sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1_score = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
        rows.append(
            f'{label:<5}  {true_positive:<2}  {false_positive:<2}  {false_negative:<2}  {true_negative:<2}  '
            f'{sensitivity:.4f}       {specificity:.4f}       {precision:.4f}   {f1_score:.4f}'
        )
    return matrix, '\n'.join(rows)


########################## testing function #########################
def testing_funnction(test_model=None, train_loader=None, val_loader=None, test_loader=None, gpu_device=0,
                      out_mode=None, weight_path=None, log_path=None):
    """Evaluate a model and append its classification report and probability-based AUC to its training log."""
    import os

    def _forward(images):
        if out_mode == 'single':
            return test_model(images)
        if out_mode == 'triplet':
            _, _, outputs = test_model(images)
            return outputs
        if out_mode == 'five':
            _, _, outputs, _, _ = test_model(images)
            return outputs
        raise ValueError(f'Unsupported out_mode: {out_mode!r}')

    def _mean_accuracy(loader):
        accuracies = []
        for images, labels in loader:
            images, labels = images.cuda(gpu_device), labels.cuda(gpu_device)
            with torch.no_grad():
                predictions = torch.argmax(_forward(images), dim=1)
            accuracies.append(accuracy_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy()))
        return float(np.mean(accuracies)) if accuracies else float('nan')

    test_model.eval()
    train_accuracy = _mean_accuracy(train_loader)
    val_accuracy = _mean_accuracy(val_loader)

    labels_all, predictions_all = [], []
    test_accuracies = []
    for images, labels in test_loader:
        images, labels = images.cuda(gpu_device), labels.cuda(gpu_device)
        with torch.no_grad():
            logits = _forward(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        labels_cpu = labels.detach().cpu().numpy()
        predictions_cpu = predictions.detach().cpu().numpy()
        test_accuracies.append(accuracy_score(labels_cpu, predictions_cpu))
        labels_all.append(labels_cpu)
        predictions_all.append(predictions_cpu)

    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(predictions_all)
    class_count = int(max(np.max(y_true), np.max(y_pred))) + 1
    y_true_one_hot = one_hot(org_x=y_true, pre_dim=class_count)
    y_pred_one_hot = one_hot(org_x=y_pred, pre_dim=class_count)
    class_report = classification_report(
        y_true, y_pred, labels=list(range(class_count)), digits=4, zero_division=0
    )
    matrix, confusion_metrics_report = confusion_matrix_metrics(
        y_true, y_pred, labels=list(range(class_count))
    )
    try:
        test_auc = float(roc_auc_score(y_true_one_hot, y_pred_one_hot))
    except ValueError as error:
        test_auc = float('nan')
        print(f'AUC unavailable: {error}')

    test_accuracy = float(np.mean(test_accuracies)) if test_accuracies else float('nan')
    print('########################## testing results #########################')
    print('train_acc:{:.4f}  val_acc:{:.4f}  test_acc:{:.4f}'.format(train_accuracy, val_accuracy, test_accuracy))
    print('########################## testing set results #########################')
    print(class_report)
    print('########################## confusion matrix #########################')
    print(matrix)
    print('########################## confusion-matrix metrics #########################')
    print(confusion_metrics_report)
    print('########################## auc results #########################')
    print('{:.4f}'.format(test_auc))

    if log_path is None and weight_path:
        log_path = os.path.splitext(weight_path)[0] + '_training_log.txt'
    if log_path:
        log_dir = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(log_dir, exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as log_file:
            log_file.write('########################## testing results #########################\n')
            log_file.write('train_acc:{:.4f}  val_acc:{:.4f}  test_acc:{:.4f}\n'.format(
                train_accuracy, val_accuracy, test_accuracy
            ))
            log_file.write('########################## testing set results #########################\n')
            log_file.write(class_report + '\n')
            log_file.write('########################## confusion matrix #########################\n')
            log_file.write(np.array2string(matrix) + '\n')
            log_file.write('########################## confusion-matrix metrics #########################\n')
            log_file.write(confusion_metrics_report + '\n')
            log_file.write('########################## auc results #########################\n')
            log_file.write('{:.4f}\n'.format(test_auc))
        print(f'Confusion-matrix metrics saved to: {log_path}')

    return {
        'test_acc': test_accuracy,
        'test_auc': test_auc,
        'classification_report': class_report,
        'confusion_matrix': matrix,
        'confusion_matrix_metrics': confusion_metrics_report,
    }

########################## single-out fitting function #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
from sklearn.metrics import accuracy_score, classification_report


def Single_out_fit(ddai_net=None, train_loader=None, val_loader=None, test_loader=None, lr_fn=None, epoch=100,
                   gpu_device=0, onecycle_mr=1e-2, current_lr=None,
                   weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1\_.pth',
                   out_mode = 'other'):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    #torch.cuda.empty_cache()
    #torch.cuda.empty_cache()

    # 新增：日志文件保存在权重文件旁边
    import os
    log_path = os.path.splitext(weight_path)[0] + '_training_log.txt'
    log_dir = os.path.dirname(os.path.abspath(log_path))
    print(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('########################## training results #########################\n')

    print('########################## training results #########################')
    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(ddai_net.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'vit':
            rmp_optim = torch.optim.AdamW(ddai_net.parameters(), lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=vit_lr_for_breast_schedule(i))
        elif lr_fn == 'onecycle':
            pass
        elif lr_fn == 'searching_best_lr':
            rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=current_lr)
        else:
            print('erorr!!!!')
            return 0

        ddai_net.train()
        for img_data, img_label in train_loader:
            img_data = img_data.cuda(gpu_device)
            img_label = img_label.cuda(gpu_device)
            if out_mode == 'non_single':
                pre_y, _, _ = ddai_net(img_data)
            else:
                pre_y = ddai_net(img_data)
            loss_value = loss_fn(pre_y, img_label)
            # loss_value = loss_dense + loss_vit
            loss_value.backward()
            if lr_fn == 'onecycle':
                scheduler.step()
            rmp_optim.step()
            rmp_optim.zero_grad()
        #print(ddai_net.w)
        ddai_net.eval()
        train_acc = []
        for train_img, train_label in train_loader:
            train_img = train_img.cuda(gpu_device)
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                if out_mode == 'non_single':
                    train_pre_y, _, _ = ddai_net(train_img)
                else:
                    train_pre_y = ddai_net(train_img)
                #train_pre_y = torch.softmax(train_pre_y, dim=1)
                train_pre_label = torch.argmax(train_pre_y, dim=1)
                train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                train_pre_label.detach().cpu().numpy()))

        val_acc = []
        for val_img, val_label in val_loader:
            val_img = val_img.cuda(gpu_device)
            val_label = val_label.cuda(gpu_device)
            with torch.no_grad():
                if out_mode == 'non_single':
                    val_pre_y, _, _ = ddai_net(val_img)
                else:
                    val_pre_y = ddai_net(val_img)
                val_loss = loss_fn(val_pre_y, val_label)
                #val_pre_y = torch.softmax(val_pre_y, dim=1)
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

        # 新增：每个 epoch 将完全相同的数值追加到 TXT
        epoch_log = ('epoch ' + str(i + 1) +
                     '  Time:{:.3}'.format(end_time - start_time) +
                     '  train_loss:{:.4}'.format(loss_value.detach().cpu().numpy()) +
                     '  train_acc:{:.4}'.format(np.mean(train_acc)) +
                     '  val_loss:{:.4}'.format(val_loss.detach().cpu().numpy()) +
                     '  val_acc:{:.4}'.format(np.mean(val_acc)))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(epoch_log + '\n')

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    ddai_net.eval()
    test_acc = []
    # 新增：用于计算完整测试集的 classification_report
    test_true_labels = []
    test_pred_labels = []
    for test_img, test_label in test_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            if out_mode == 'non_single':
                test_pre_y, _, _ = ddai_net(test_img)
            else:
                test_pre_y = ddai_net(test_img)
            test_loss = loss_fn(test_pre_y, test_label)
            #test_pre_y = torch.softmax(test_pre_y, dim=1)
            test_pre_label = torch.argmax(test_pre_y, dim=1)
            test_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                           test_pre_label.detach().cpu().numpy()))
            # 新增：收集真实标签和预测标签，不影响原有 test_acc
            test_true_labels.extend(test_label.detach().cpu().numpy().tolist())
            test_pred_labels.extend(test_pre_label.detach().cpu().numpy().tolist())

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))

    # 新增：训练结束，在 TXT 最后写入测试结果
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('########################## testing results #########################\n')
        f.write('train_acc:{:.4}'.format(np.mean(train_acc)) +
                '  val_acc:{:.4}'.format(np.mean(val_acc)) +
                '  test_acc:{:.4}'.format(np.mean(test_acc)) + '\n')

    # 新增：打印并追加保存 sklearn 的分类报告
    class_report = classification_report(test_true_labels, test_pred_labels, digits=4, zero_division=0)
    print('########################## classification report #########################')
    print(class_report)
    test_true_array = np.asarray(test_true_labels, dtype=int)
    test_pred_array = np.asarray(test_pred_labels, dtype=int)
    class_count = int(max(np.max(test_true_array), np.max(test_pred_array))) + 1
    matrix, confusion_metrics_report = confusion_matrix_metrics(
        test_true_array, test_pred_array, labels=list(range(class_count))
    )
    test_true_one_hot = one_hot(org_x=test_true_array, pre_dim=class_count)
    test_pred_one_hot = one_hot(org_x=test_pred_array, pre_dim=class_count)
    try:
        test_auc = float(roc_auc_score(test_true_one_hot, test_pred_one_hot))
    except ValueError as error:
        test_auc = float('nan')
        print(f'AUC unavailable: {error}')

    print('########################## confusion matrix #########################')
    print(matrix)
    print('########################## confusion-matrix metrics #########################')
    print(confusion_metrics_report)
    print('########################## auc results #########################')
    print('{:.4f}'.format(test_auc))
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('########################## classification report #########################\n')
        f.write(class_report + '\n')
        f.write('########################## confusion matrix #########################\n')
        f.write(np.array2string(matrix) + '\n')
        f.write('########################## confusion-matrix metrics #########################\n')
        f.write(confusion_metrics_report + '\n')
        f.write('########################## auc results #########################\n')
        f.write('{:.4f}\n'.format(test_auc))

    g = ddai_net.state_dict()
    torch.save(g, weight_path)

    return test_acc


########################## double-out fitting function #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
def Multiple_out_fit(ddai_net = None, train_loader = None, val_loader = None, test_loader = None,
                   epoch = 100, gpu_device = 0):
    ViT_loss = nn.CrossEntropyLoss()
    FABNet_loss = nn.CrossEntropyLoss()
    ddai_net = ddai_net.cuda(gpu_device)

    for i in range(epoch):
        start_time = time.time()
        rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=vit_lr_schedule(i))
        ddai_net.train()
        for img_data, img_label in train_loader:
            img_data = img_data.cuda(gpu_device)
            img_label = img_label.cuda(gpu_device)
            vit_y, dense_y = ddai_net(img_data)
            loss_ViT = ViT_loss(vit_y, img_label)
            loss_FABNet = FABNet_loss(dense_y, img_label)
            # loss_value = loss_dense + loss_vit
            loss_ViT.backward()
            loss_FABNet.backward()
            # scheduler.step()
            rmp_vit_optim.step()
            rmp_vit_optim.zero_grad()
            rmp_fab_optim.step()
            rmp_fab_optim.zero_grad()

        ddai_net.eval()
        train_FAB_acc = []
        train_ViT_acc = []
        for train_img, train_label in train_loader:
            train_img = train_img.cuda(gpu_device)
            train_label = train_label.cuda(gpu_device)
            with torch.no_grad():
                train_ViT_y, train_FAB_y = ddai_net(train_img)
                pre_train_FAB = torch.argmax(train_FAB_y, dim=1)
                pre_train_ViT = torch.argmax(train_ViT_y, dim=1)
                train_FAB_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                    pre_train_FAB.detach().cpu().numpy()))
                train_ViT_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                    pre_train_ViT.detach().cpu().numpy()))

        val_FAB_acc = []
        val_ViT_acc = []
        for val_img, val_label in val_loader:
            val_img = val_img.cuda(gpu_device)
            val_label = val_label.cuda(gpu_device)
            with torch.no_grad():
                val_ViT_y, val_FAB_y = ddai_net(val_img)
                val_FAB_loss = FABNet_loss(val_FAB_y, val_label)
                val_ViT_loss = ViT_loss(val_ViT_y, val_label)
                pre_val_FAB = torch.argmax(val_FAB_y, dim=1)
                pre_val_ViT = torch.argmax(val_ViT_y, dim=1)
                val_FAB_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                                  pre_val_FAB.detach().cpu().numpy()))
                val_ViT_acc.append(accuracy_score(val_label.detach().cpu().numpy(),
                                                  pre_val_ViT.detach().cpu().numpy()))

        end_time = time.time()
        print('epoch ' + str(i + 1), ' train_FAB_loss:{:.4}'.format(loss_FABNet.detach().cpu().numpy()),
              ' train_FAB_acc:{:.4}'.format(np.mean(train_FAB_acc)),
              ' val_FAB_loss:{:.4}'.format(val_FAB_loss.detach().cpu().numpy()),
              ' val_FAB_acc:{:.4}'.format(np.mean(val_FAB_acc)),
              ' Time:{:.3}'.format(end_time - start_time), '\n',
              '________train_ViT_loss:{:.4}'.format(loss_ViT.detach().cpu().numpy()),
              ' train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
              '   val_ViT_loss:{:.4}'.format(val_ViT_loss.detach().cpu().numpy()),
              ' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    ddai_net.eval()
    test_FAB_acc = []
    test_ViT_acc = []
    for test_img, test_label in test_loader:
        test_img = test_img.cuda(gpu_device)
        test_label = test_label.cuda(gpu_device)
        with torch.no_grad():
            test_ViT_y, test_FAB_y = ddai_net(test_img)
            test_FAB_loss = FABNet_loss(test_FAB_y, test_label)
            pre_test_FAB = torch.argmax(test_FAB_y, dim=1)
            test_FAB_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               pre_test_FAB.detach().cpu().numpy()))

            test_ViT_loss = ViT_loss(test_ViT_y, test_label)
            pre_test_ViT = torch.argmax(test_ViT_y, dim=1)
            test_ViT_acc.append(accuracy_score(test_label.detach().cpu().numpy(),
                                               pre_test_ViT.detach().cpu().numpy()))

    print('train_FAB_acc:{:.4}'.format(np.mean(train_FAB_acc)),
          ' val_FAB_acc:{:.4}'.format(np.mean(val_FAB_acc)),
          ' test_FAB_acc:{:.4}'.format(np.mean(test_FAB_acc)), '\n',
          'train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
          ' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)),
          ' test_ViT_acc:{:.4}'.format(np.mean(test_ViT_acc)))
    # write_1.add_graph(ddai_net, input_to_model=train_img[0:2])
    g = ddai_net.state_dict()
    torch.save(g, 'E:\SOTA_Model_Classification\SC_Weights\Larynx\SwinT_EVD.pth')

########################## multiple_show_acc_loss function #########################
def multiple_show_acc_loss(ddai_net = None, data_loader = None, gpu_device=0):
    ddai_net.eval()
    train_FAB_acc = []
    train_ViT_acc = []
    sp_3_acc = []
    sp_4_acc = []
    loss_1 = []
    loss_2 = []
    loss_3 = []
    loss_fn = nn.CrossEntropyLoss()
    for train_img, train_label in data_loader:
        train_img = train_img.cuda(gpu_device)
        train_label = train_label.cuda(gpu_device)
        with torch.no_grad():
            train_ViT_y, train_FAB_y, sp_3_y = ddai_net(train_img)
            pre_train_FAB = torch.argmax(train_FAB_y, dim=1)
            pre_train_ViT = torch.argmax(train_ViT_y, dim=1)
            pre_sp3_y = torch.argmax(sp_3_y, dim=1)
            train_FAB_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                pre_train_FAB.detach().cpu().numpy()))
            train_ViT_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                pre_train_ViT.detach().cpu().numpy()))
            sp_3_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                                pre_sp3_y.detach().cpu().numpy()))
            loss_1.append(loss_fn(train_ViT_y, train_label).detach().cpu().numpy())
            loss_2.append(loss_fn(train_FAB_y, train_label).detach().cpu().numpy())
            loss_3.append(loss_fn(sp_3_y, train_label).detach().cpu().numpy())
    return train_ViT_acc, train_FAB_acc, sp_3_acc, loss_1, loss_2, loss_3


########################## Multiple_out_fit function #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
#### return:
def Multiple_out_fit(ddai_net = None, train_loader = None, val_loader = None, test_loader = None,
                   epoch = 100, gpu_device = 0, weight_path = None):
    sp_1_loss = nn.CrossEntropyLoss()
    sp_2_loss = nn.CrossEntropyLoss()
    ddai_net = ddai_net.cuda(gpu_device)

    for i in range(epoch):
        start_time = time.time()
        rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=vit_lr_schedule(i))
        ddai_net.train()

        for img_data, img_label in train_loader:
            img_data = img_data.cuda(gpu_device)
            img_label = img_label.cuda(gpu_device)
            sp_1_y, sp_2_y, sp_3_y = ddai_net(img_data)
            loss_sp_1 = sp_2_loss(sp_1_y, img_label)
            loss_sp_2 = sp_2_loss(sp_2_y, img_label)
            loss_sp_3 = sp_2_loss(sp_3_y, img_label)
            loss_value = loss_sp_1 + loss_sp_2 + loss_sp_3
            loss_value.backward()
            rmp_optim.step()
            rmp_optim.zero_grad()

        sp_1_train_acc, sp_2_train_acc,sp_3_train_acc,train_loss_1, train_loss_2, \
        train_loss_3 = multiple_show_acc_loss(ddai_net = ddai_net, data_loader = train_loader, gpu_device=gpu_device)
        sp_1_val_acc, sp_2_val_acc, sp_3_val_acc,val_loss_1, val_loss_2, val_loss_3 = \
            multiple_show_acc_loss(ddai_net = ddai_net, data_loader = val_loader, gpu_device=gpu_device)

        end_time = time.time()
        print('epoch ' + str(i + 1), ' train_loss_1:{:.4}'.format(np.mean(train_loss_1)),
              ' sp_1_train_acc:{:.4}'.format(np.mean(sp_1_train_acc)),
              ' val_loss_1:{:.4}'.format(np.mean(val_loss_1)),
              ' sp_1_val_acc:{:.4}'.format(np.mean(sp_1_val_acc)),
              ' Time:{:.3}'.format(end_time - start_time), '\n',
              ' _________train_loss_2:{:.4}'.format(np.mean(train_loss_2)),
              ' sp_2_train_acc:{:.4}'.format(np.mean(sp_2_train_acc)),
              ' val_loss_2:{:.4}'.format(np.mean(val_loss_2)),
              ' sp_2_val_acc:{:.4}'.format(np.mean(sp_2_val_acc)), '\n',
              ' _________train_loss_3:{:.4}'.format(np.mean(train_loss_3)),
              ' sp_3_train_acc:{:.4}'.format(np.mean(sp_3_train_acc)),
              ' val_loss_3:{:.4}'.format(np.mean(val_loss_3)),
              ' sp_3_val_acc:{:.4}'.format(np.mean(sp_3_val_acc)))

    sp_1_test_acc, sp_2_test_acc, sp_3_test_acc, test_loss_1, test_loss_2, test_loss_3 = \
        multiple_show_acc_loss(ddai_net=ddai_net, data_loader=test_loader, gpu_device=gpu_device)

    print('sp_1_train_acc:{:.4}'.format(np.mean(sp_1_train_acc)),
          ' sp_1_val_acc:{:.4}'.format(np.mean(sp_1_val_acc)),
          ' sp_1_test_acc:{:.4}'.format(np.mean(sp_1_test_acc)), '\n',
          ' sp_2_train_acc:{:.4}'.format(np.mean(sp_2_train_acc)),
          ' sp_2_val_acc:{:.4}'.format(np.mean(sp_2_val_acc)),
          ' sp_2_test_acc:{:.4}'.format(np.mean(sp_2_test_acc)), '\n',
          ' sp_3_train_acc:{:.4}'.format(np.mean(sp_3_train_acc)),
          ' sp_3_val_acc:{:.4}'.format(np.mean(sp_3_val_acc)),
          ' sp_3_test_acc:{:.4}'.format(np.mean(sp_3_test_acc)))
    # write_1.add_graph(ddai_net, input_to_model=train_img[0:2])
    g = ddai_net.state_dict()
    torch.save(g, weight_path)


########################## The function for searching the best learning rates #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
#### return:
def searching_best_lr(min_lr = 1e-7, max_lr = 1e-3, fitting_model = 'single_out', class_num=None,
                        train_loader=None, val_loader=None, test_loader=None, file_name='searching_best_lr.csv',
                      lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-5):
    testing_lr_acc = []
    max_boundary = -np.log10(max_lr)
    min_boundary = -np.log10(min_lr)

    change_log = 0
    for k in range(int((min_boundary - max_boundary) * 10)):
        ########################## xxxxxx #########################
        setup_seed(0)

        #vit_base = creating_ViT(class_num = class_num, gpu_device=gpu_device)
        #swinT_base = creating_swinT(class_num=class_num, gpu_device=gpu_device)
        mlp_mixer_base = creating_Mixer(class_num=class_num, gpu_device=gpu_device)
        #print(vit_base.head.weight)

        ########################## xxxxxx #########################
        if k % 10 == 0:
            change_log += 1
        current_lr = (max_lr / (10 ** (change_log - 1))) - (k - (change_log - 1) * 10) * \
                     (max_lr / (10 ** change_log))
        print('current_lr:', current_lr)
        if fitting_model == 'single_out':
            test_acc = Single_out_fit(ddai_net=mlp_mixer_base, train_loader=train_loader, val_loader=val_loader,
                           test_loader=test_loader, lr_fn=lr_fn, epoch=epoch, gpu_device=gpu_device,
                           onecycle_mr=onecycle_mr, current_lr=current_lr)
        else:
            pass
        testing_lr_acc.append([current_lr, np.mean(test_acc)])
        #del vit_net
        #del vit_base
    testing_lr_acc = np.array(testing_lr_acc).reshape((int((min_boundary - max_boundary) * 10), 2))
    testing_lr_acc_pd = pd.DataFrame(testing_lr_acc)
    testing_lr_acc_pd.to_csv('E:\SOTA_Model_Classification\SC_Results\SwinT\\' + file_name)
    print(testing_lr_acc)
    del swinT_base
    return testing_lr_acc

if __name__ == '__main__':
    max_lr = 1e-3
    min_lr = 1e-7
    max_boundary = -np.log10(max_lr)
    min_boundary = -np.log10(min_lr)

    change_log = 0
    for k in range(int(min_boundary - max_boundary) * 10):
        if k % 10 == 0:
            change_log += 1
        lr = (max_lr / (10**(change_log - 1))) - (k - (change_log-1) * 10) * (max_lr / (10**change_log))
        print(lr)





