import torch
from torch import nn
import time
import contextlib
import sys
from datetime import datetime
from pathlib import Path
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



class _ConsoleAndFile:
    """File-like stream that preserves terminal output while persisting it."""
    def __init__(self, terminal, file_handle):
        self.terminal = terminal
        self.file_handle = file_handle

    def write(self, message):
        self.terminal.write(message)
        self.file_handle.write(message)

    def flush(self):
        self.terminal.flush()
        self.file_handle.flush()


def testing_fusion_function(test_model=None, train_loader=None, val_loader=None, test_loader=None,
                            gpu_device=0, out_mode='triplet', weight_path=None):
    """Run fusion evaluation and append all printed metrics to its weight-adjacent TXT log."""
    if weight_path is None:
        raise ValueError('weight_path must be provided for fusion test logging.')
    log_path = Path(weight_path).with_suffix('.txt')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Fusion test log: {log_path}', flush=True)
    with log_path.open('a', encoding='utf-8') as log_file:
        log_file.write(f'\\n########################## Fusion test run {datetime.now().isoformat(timespec="seconds")} #########################\\n')
        log_file.write(f'Weight path: {weight_path}\\n')
        tee = _ConsoleAndFile(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee):
            return testing_funnction(
                test_model=test_model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                gpu_device=gpu_device,
                out_mode=out_mode,
            )



class _FusionTrainingLogger:
    """Mirror fusion-training progress to the terminal and a weight-adjacent text file."""
    def __init__(self, weight_path, name):
        self.path = Path(weight_path).with_suffix('.txt')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open('w', encoding='utf-8')
        print(f'Fusion training log: {self.path}', flush=True)
        self.log(f'Fusion method: {name}')
        self.log(f'Started: {datetime.now().isoformat(timespec="seconds")}')
        self.log(f'Weight path: {weight_path}')

    def log(self, message):
        print(message)
        self._file.write(str(message) + '\n')
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self.log(f'Finished: {datetime.now().isoformat(timespec="seconds")}')
            self._file.close()




def _log_fusion_test_classification_report(logger, true_labels, predicted_labels):
    """Append the complete sklearn classification report for a fusion test run."""
    if not true_labels:
        logger.log('########################## sklearn classification report (test) #########################')
        logger.log('No test samples were available.')
        return
    report = classification_report(true_labels, predicted_labels, digits=4, zero_division=0)
    logger.log('########################## sklearn classification report (test) #########################')
    logger.log(report)
def train_concatenation_fusion(model=None, train_loader=None, val_loader=None, test_loader=None,
                                epoch=100, gpu_device=0, weight_path=None):
    """Train the feature-concatenation baseline and save only the final-epoch model."""
    if model is None:
        raise ValueError('model must be provided.')
    if weight_path is None:
        raise ValueError('weight_path must be provided.')

    device = gpu_device if isinstance(gpu_device, torch.device) else torch.device(f'cuda:{gpu_device}')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.swin_backbone.parameters(), 'lr': 1e-5},
        {'params': model.vig_stem.parameters(), 'lr': 1e-5},
        {'params': model.vig_blocks.parameters(), 'lr': 1e-5},
        {'params': model.vig_projection.parameters(), 'lr': 1e-5},
        {'params': model.vig_head.parameters(), 'lr': 1e-5},
        {'params': model.fusion_classifier.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    logger = _FusionTrainingLogger(weight_path, 'Concatenation Fusion')
    logger.log('########################## Concatenation Fusion training results #########################')

    for current_epoch in range(epoch):
        start_time = time.time()
        # Preserve optimizer moments while applying the existing three-stage schedule.
        backbone_lr = vit_lr_schedule(current_epoch)
        fusion_lr = fusion_lr_schedule(current_epoch)
        for group_index, parameter_group in enumerate(optimizer.param_groups):
            parameter_group['lr'] = fusion_lr if group_index == len(optimizer.param_groups) - 1 else backbone_lr

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, _, fusion_logits = model(images)
            loss = criterion(fusion_logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (fusion_logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, _, fusion_logits = model(images)
                loss = criterion(fusion_logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (fusion_logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        elapsed = time.time() - start_time
        logger.log(f'epoch {current_epoch + 1} Time:{elapsed:.1f} '
              f'train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} '
              f'val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}')



    model.eval()
    test_correct = 0
    test_total = 0
    test_true_labels = []
    test_predicted_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, _, fusion_logits = model(images)
            predictions = fusion_logits.argmax(dim=1)
            test_correct += (predictions == labels).sum().item()
            test_true_labels.extend(labels.detach().cpu().tolist())
            test_predicted_labels.extend(predictions.detach().cpu().tolist())
            test_total += labels.size(0)
    test_acc = test_correct / max(test_total, 1)
    torch.save(model.state_dict(), weight_path)
    logger.log('########################## Concatenation Fusion testing results #########################')
    logger.log(f'last_epoch_val_acc:{val_acc:.4f} test_acc:{test_acc:.4f}')
    _log_fusion_test_classification_report(logger, test_true_labels, test_predicted_labels)
    logger.close()
    return test_acc


def train_cross_attention_fusion(model=None, train_loader=None, val_loader=None, test_loader=None,
                                 epoch=100, gpu_device=0, weight_path=None):
    """Train Cross_Attetion_Fusion and save the best validation checkpoint."""
    if model is None:
        raise ValueError('model must be provided.')
    if weight_path is None:
        raise ValueError('weight_path must be provided.')

    device = gpu_device if isinstance(gpu_device, torch.device) else torch.device(f'cuda:{gpu_device}')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.swin_backbone.parameters(), 'lr': 1e-5},
        {'params': model.vig_stem.parameters(), 'lr': 1e-5},
        {'params': model.vig_blocks.parameters(), 'lr': 1e-5},
        {'params': model.vig_projection.parameters(), 'lr': 1e-5},
        {'params': model.vig_head.parameters(), 'lr': 1e-5},
        {'params': model.vig_to_swin.parameters(), 'lr': 5e-5},
        {'params': model.cross_attention.parameters(), 'lr': 5e-5},
        {'params': model.cross_norm.parameters(), 'lr': 5e-5},
        {'params': model.fusion_classifier.parameters(), 'lr': 5e-5},
    ], weight_decay=1e-4)

    logger = _FusionTrainingLogger(weight_path, 'Cross Attention Fusion')
    logger.log('########################## Cross Attention Fusion training results #########################')
    for current_epoch in range(epoch):
        start_time = time.time()
        backbone_lr = vit_lr_schedule(current_epoch)
        fusion_lr = fusion_lr_schedule(current_epoch)
        for index, group in enumerate(optimizer.param_groups):
            group['lr'] = backbone_lr if index < 5 else fusion_lr

        model.train()
        train_loss_sum = train_correct = train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, _, logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss_sum = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, _, logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        logger.log(f'epoch {current_epoch + 1} Time:{time.time() - start_time:.1f} '
              f'train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} '
              f'val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}')

    model.eval()
    test_correct = test_total = 0
    test_true_labels = []
    test_predicted_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, _, logits = model(images)
            predictions = logits.argmax(dim=1)
            test_correct += (predictions == labels).sum().item()
            test_true_labels.extend(labels.detach().cpu().tolist())
            test_predicted_labels.extend(predictions.detach().cpu().tolist())
            test_total += labels.size(0)
    test_acc = test_correct / max(test_total, 1)
    torch.save(model.state_dict(), weight_path)
    logger.log('########################## Cross Attention Fusion testing results #########################')
    logger.log(f'last_epoch_val_acc:{val_acc:.4f} test_acc:{test_acc:.4f}')
    _log_fusion_test_classification_report(logger, test_true_labels, test_predicted_labels)
    logger.close()
    return test_acc


def train_late_fusion(model=None, train_loader=None, val_loader=None, test_loader=None,
                      epoch=100, gpu_device=0, weight_path=None):
    """Train the logit-level Late_Fusion baseline and save only its final-epoch checkpoint."""
    if model is None:
        raise ValueError('model must be provided.')
    if weight_path is None:
        raise ValueError('weight_path must be provided.')

    device = gpu_device if isinstance(gpu_device, torch.device) else torch.device(f'cuda:{gpu_device}')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.swin_backbone.parameters(), 'lr': 1e-5},
        {'params': model.vig_stem.parameters(), 'lr': 1e-5},
        {'params': model.vig_blocks.parameters(), 'lr': 1e-5},
        {'params': model.vig_projection.parameters(), 'lr': 1e-5},
        {'params': model.vig_head.parameters(), 'lr': 1e-5},
        {'params': [model.fusion_weights], 'lr': 1e-4},
    ], weight_decay=1e-4)

    logger = _FusionTrainingLogger(weight_path, 'Late Fusion')
    logger.log('########################## Late Fusion training results #########################')
    for current_epoch in range(epoch):
        start_time = time.time()
        backbone_lr = vit_lr_schedule(current_epoch)
        fusion_lr = fusion_lr_schedule(current_epoch)
        for index, group in enumerate(optimizer.param_groups):
            group['lr'] = backbone_lr if index < 5 else fusion_lr

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, _, logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, _, logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        weights = torch.softmax(model.fusion_weights.detach(), dim=0).cpu().tolist()
        logger.log(f'epoch {current_epoch + 1} Time:{time.time() - start_time:.1f} '
              f'train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} '
              f'val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} '
              f'swin_weight:{weights[0]:.4f} vig_weight:{weights[1]:.4f}')

    model.eval()
    test_correct = 0
    test_total = 0
    test_true_labels = []
    test_predicted_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, _, logits = model(images)
            predictions = logits.argmax(dim=1)
            test_correct += (predictions == labels).sum().item()
            test_true_labels.extend(labels.detach().cpu().tolist())
            test_predicted_labels.extend(predictions.detach().cpu().tolist())
            test_total += labels.size(0)
    test_acc = test_correct / max(test_total, 1)
    torch.save(model.state_dict(), weight_path)
    weights = torch.softmax(model.fusion_weights.detach(), dim=0).cpu().tolist()
    logger.log('########################## Late Fusion testing results #########################')
    logger.log(f'last_epoch_val_acc:{val_acc:.4f} test_acc:{test_acc:.4f} '
          f'swin_weight:{weights[0]:.4f} vig_weight:{weights[1]:.4f}')
    _log_fusion_test_classification_report(logger, test_true_labels, test_predicted_labels)
    logger.close()
    return test_acc


def train_weighted_summation(model=None, train_loader=None, val_loader=None, test_loader=None,
                              epoch=100, gpu_device=0, weight_path=None):
    """Train the feature-level Weighted_Summation baseline and save only its final-epoch checkpoint."""
    if model is None:
        raise ValueError('model must be provided.')
    if weight_path is None:
        raise ValueError('weight_path must be provided.')

    device = gpu_device if isinstance(gpu_device, torch.device) else torch.device(f'cuda:{gpu_device}')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.swin_backbone.parameters(), 'lr': 1e-5},
        {'params': model.vig_stem.parameters(), 'lr': 1e-5},
        {'params': model.vig_blocks.parameters(), 'lr': 1e-5},
        {'params': model.vig_projection.parameters(), 'lr': 1e-5},
        {'params': model.vig_head.parameters(), 'lr': 1e-5},
        {'params': model.swin_projector.parameters(), 'lr': 1e-4},
        {'params': model.vig_projector.parameters(), 'lr': 1e-4},
        {'params': [model.fusion_weights], 'lr': 1e-4},
        {'params': model.fusion_classifier.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    logger = _FusionTrainingLogger(weight_path, 'Weighted Summation')
    logger.log('########################## Weighted Summation training results #########################')
    for current_epoch in range(epoch):
        start_time = time.time()
        backbone_lr = vit_lr_schedule(current_epoch)
        fusion_lr = fusion_lr_schedule(current_epoch)
        for index, group in enumerate(optimizer.param_groups):
            group['lr'] = backbone_lr if index < 5 else fusion_lr

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, _, logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, _, logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        weights = torch.softmax(model.fusion_weights.detach(), dim=0).cpu().tolist()
        logger.log(f'epoch {current_epoch + 1} Time:{time.time() - start_time:.1f} '
              f'train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} '
              f'val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} '
              f'swin_weight:{weights[0]:.4f} vig_weight:{weights[1]:.4f}')

    model.eval()
    test_correct = 0
    test_total = 0
    test_true_labels = []
    test_predicted_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, _, logits = model(images)
            predictions = logits.argmax(dim=1)
            test_correct += (predictions == labels).sum().item()
            test_true_labels.extend(labels.detach().cpu().tolist())
            test_predicted_labels.extend(predictions.detach().cpu().tolist())
            test_total += labels.size(0)
    test_acc = test_correct / max(test_total, 1)
    torch.save(model.state_dict(), weight_path)
    weights = torch.softmax(model.fusion_weights.detach(), dim=0).cpu().tolist()
    logger.log('########################## Weighted Summation testing results #########################')
    logger.log(f'last_epoch_val_acc:{val_acc:.4f} test_acc:{test_acc:.4f} '
          f'swin_weight:{weights[0]:.4f} vig_weight:{weights[1]:.4f}')
    _log_fusion_test_classification_report(logger, test_true_labels, test_predicted_labels)
    logger.close()
    return test_acc


def train_gated_fusion(model=None, train_loader=None, val_loader=None, test_loader=None,
                       epoch=100, gpu_device=0, weight_path=None):
    """Train the sample-adaptive Gated_Fusion baseline and save only its final-epoch checkpoint."""
    if model is None:
        raise ValueError('model must be provided.')
    if weight_path is None:
        raise ValueError('weight_path must be provided.')

    device = gpu_device if isinstance(gpu_device, torch.device) else torch.device(f'cuda:{gpu_device}')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.swin_backbone.parameters(), 'lr': 1e-5},
        {'params': model.vig_stem.parameters(), 'lr': 1e-5},
        {'params': model.vig_blocks.parameters(), 'lr': 1e-5},
        {'params': model.vig_projection.parameters(), 'lr': 1e-5},
        {'params': model.vig_head.parameters(), 'lr': 1e-5},
        {'params': model.swin_projector.parameters(), 'lr': 1e-4},
        {'params': model.vig_projector.parameters(), 'lr': 1e-4},
        {'params': model.gate_network.parameters(), 'lr': 1e-4},
        {'params': model.fusion_classifier.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    logger = _FusionTrainingLogger(weight_path, 'Gated Fusion')
    logger.log('########################## Gated Fusion training results #########################')
    for current_epoch in range(epoch):
        start_time = time.time()
        backbone_lr = vit_lr_schedule(current_epoch)
        fusion_lr = fusion_lr_schedule(current_epoch)
        for index, group in enumerate(optimizer.param_groups):
            group['lr'] = backbone_lr if index < 5 else fusion_lr

        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        gate_sum = 0.0
        gate_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, _, logits, gate = model(images, return_gate=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
            gate_sum += gate.detach().sum().item()
            gate_count += gate.numel()

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, _, logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        mean_swin_gate = gate_sum / max(gate_count, 1)
        logger.log(f'epoch {current_epoch + 1} Time:{time.time() - start_time:.1f} '
                   f'train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} '
                   f'val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} '
                   f'mean_swin_gate:{mean_swin_gate:.4f} mean_vig_gate:{1.0 - mean_swin_gate:.4f}')

    model.eval()
    test_correct = 0
    test_total = 0
    test_true_labels = []
    test_predicted_labels = []
    gate_sum = 0.0
    gate_count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, _, logits, gate = model(images, return_gate=True)
            predictions = logits.argmax(dim=1)
            test_correct += (predictions == labels).sum().item()
            test_true_labels.extend(labels.detach().cpu().tolist())
            test_predicted_labels.extend(predictions.detach().cpu().tolist())
            test_total += labels.size(0)
            gate_sum += gate.sum().item()
            gate_count += gate.numel()
    test_acc = test_correct / max(test_total, 1)
    torch.save(model.state_dict(), weight_path)
    mean_swin_gate = gate_sum / max(gate_count, 1)
    logger.log('########################## Gated Fusion testing results #########################')
    logger.log(f'last_epoch_val_acc:{val_acc:.4f} test_acc:{test_acc:.4f} '
               f'mean_swin_gate:{mean_swin_gate:.4f} mean_vig_gate:{1.0 - mean_swin_gate:.4f}')
    _log_fusion_test_classification_report(logger, test_true_labels, test_predicted_labels)
    logger.close()
    return test_acc

