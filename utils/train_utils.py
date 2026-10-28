import os
import torch
import numpy as np
import random
import sys
import time
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, accuracy_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score
import torch.nn.functional as F


def confusion_matrix_metrics(y_true, y_pred, labels):
	"""Return the confusion matrix and per-class metrics derived from it."""
	matrix = confusion_matrix(y_true, y_pred, labels=labels)
	total = int(matrix.sum())
	rows = ['Class  TP  FP  FN  TN  Sensitivity  Specificity  Precision  F1-score']
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
from loss_funcs import adv, daan, coral, bnm, mmd, lmmd
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F

def train_single(net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None):
	net = net.cuda(gpu_device)
	loss_cls = nn.CrossEntropyLoss()
	for i in range(epoch):
		start_time = time.time()
		optim = torch.optim.AdamW(net.parameters(), vit_lr_schedule(i))
		net.train()
		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label =\
				img_label.cuda(gpu_device)
			pre_y = net(img_data)
			loss = loss_cls(pre_y,img_label)
			loss.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_pre_y = net(train_img)
				#print(train_pre_y)
				train_pre_y = torch.argmax(train_pre_y, dim=1)
				train_acc.append(accuracy_score(train_label.detach().cpu().numpy(), train_pre_y.detach().cpu().numpy()))


		val_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_pre_y = net(val_img)
				val_cls_loss = loss_cls(val_pre_y, val_label)
				val_pre_y = torch.argmax(val_pre_y, dim=1)
				val_acc.append(accuracy_score(val_label.detach().cpu().numpy(), val_pre_y.detach().cpu().numpy()))

		end_time = time.time()
		#print(net.patch_embed.proj.weight)
		print('epoch ' + str(i + 1), ' Time:{:.3}'.format(end_time - start_time), '\n',
			  ' ________train_loss:{:.4}'.format(loss.detach().cpu().numpy()),
			  ' train_acc:{:.4}'.format(np.mean(train_acc)), '\n',
			  ' ________  val_loss:{:.4}'.format(val_cls_loss.detach().cpu().numpy()),
			  '   val_acc:{:.4}'.format(np.mean(val_acc)))

	net.eval()
	test_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_pre_y = net(test_img)
			test_loss = loss_cls(test_pre_y, test_label)
			test_pre_y = torch.argmax(test_pre_y, dim=1)
			test_acc.append(accuracy_score(test_label.detach().cpu().numpy(), test_pre_y.detach().cpu().numpy()))
	print('######################test_result###########################')
	print('train_acc:{:.4}'.format(np.mean(train_acc)),
		  ' val_acc:{:.4}'.format(np.mean(val_acc)),
		  ' test_acc:{:.4}'.format(np.mean(test_acc)), '\n',)

	g = net.state_dict()
	torch.save(g, weight_path)

	return test_acc


def train_baseline(number=None, fusion_net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None):
	fusion_net = fusion_net.to(gpu_device)
	Loss_SwinT = nn.CrossEntropyLoss()
	Loss_ViG = nn.CrossEntropyLoss()
	Loss_Fusion = nn.CrossEntropyLoss()

	for i in range(epoch):
		start_time = time.time()
		optim = torch.optim.AdamW(fusion_net.parameters(), vit_lr_schedule(i))
		fusion_net.train()
		for img_data, img_label in train_loader:
			img_data = img_data.to(gpu_device)
			img_label = img_label.to(gpu_device)

			swint_y, vig_y, fusion_y = fusion_net(img_data)

			loss_SwinT = Loss_SwinT(swint_y, img_label)
			loss_ViG = Loss_ViG(vig_y, img_label)
			loss_Fusion = Loss_Fusion(fusion_y, img_label)

			loss = loss_Fusion

			loss.backward()
			optim.step()
			optim.zero_grad()

		fusion_net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []

		for train_img, train_label in train_loader:
			train_img = train_img.to(gpu_device)
			train_label = train_label.to(gpu_device)
			with torch.no_grad():
				#print(train_label)
				train_SwinT_y, train_ViG_y, train_Fusion_y = fusion_net(train_img)
				#print(train_SwinT_y)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.to(gpu_device)
			val_label = val_label.to(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y = fusion_net(val_img)
				val_SwinT_loss = Loss_SwinT(val_SwinT_y, val_label)
				val_ViG_loss = Loss_ViG(val_ViG_y, val_label)
				val_Fusion_loss = Loss_Fusion(val_Fusion_y, val_label)
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number),'\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()))

	fusion_net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.to(gpu_device)
		test_label = test_label.to(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y = fusion_net(test_img)
			test_SwinT_loss = Loss_SwinT(test_SwinT_y, test_label)
			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))
			test_VMamba_loss = Loss_ViG(test_ViG_y, test_label)
			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))
			test_Fusion_loss = Loss_Fusion(test_Fusion_y, test_label)
			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('########################## testing results #########################''\n',)
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
			'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
			' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
			'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
			' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = fusion_net.state_dict()
	torch.save(g, weight_path)
	return test_Fusion_acc

def train_baseline_w(number=None, fusion_net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
					 fusion_weight=None):
	fusion_net = fusion_net.to(gpu_device)
	Loss_SwinT = nn.CrossEntropyLoss()
	Loss_ViG = nn.CrossEntropyLoss()
	Loss_Fusion = nn.CrossEntropyLoss()

	for i in range(epoch):
		start_time = time.time()
		optim = torch.optim.AdamW(fusion_net.parameters(), cnn_lr_schedule(i))
		fusion_net.train()
		for img_data, img_label in train_loader:
			img_data = img_data.to(gpu_device)
			img_label = img_label.to(gpu_device)

			swint_y, vig_y, fusion_y, swin_feature, vig_feature = fusion_net(img_data)
			swin_feature = swin_feature.squeeze(-1)
			_, _, v1 = torch.linalg.svd(swin_feature)
			swin_feature = swin_feature @ v1[:10, :].mT
			_, _, v2 = torch.linalg.svd(vig_feature)
			vig_feature = vig_feature @ v2[:10, :].mT

			model1 = LinearRegression()
			model1.fit(swin_feature.detach().cpu().numpy(), img_label.detach().cpu().numpy())
			R1_squared = model1.score(swin_feature.detach().cpu().numpy(), img_label.detach().cpu().numpy())

			model2 = LinearRegression()
			model2.fit(vig_feature.detach().cpu().numpy(), img_label.detach().cpu().numpy())
			R2_squared = model2.score(vig_feature.detach().cpu().numpy(), img_label.detach().cpu().numpy())

			swin_weight = torch.tensor(R1_squared / (R1_squared + R2_squared))
			vig_weight = torch.tensor(R2_squared / (R1_squared + R2_squared))
			#print(swin_weight, vig_weight)
			loss_SwinT = Loss_SwinT(swint_y, img_label)
			loss_ViG = Loss_ViG(vig_y, img_label)
			loss_Fusion = Loss_Fusion(fusion_y, img_label)

			loss = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vig_weight * loss_ViG

			loss.backward()
			optim.step()
			optim.zero_grad()

		fusion_net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []

		for train_img, train_label in train_loader:
			train_img = train_img.to(gpu_device)
			train_label = train_label.to(gpu_device)
			with torch.no_grad():
				#print(train_label)
				train_SwinT_y, train_ViG_y, train_Fusion_y,_,_ = fusion_net(train_img)
				#print(train_SwinT_y)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.to(gpu_device)
			val_label = val_label.to(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y,_,_ = fusion_net(val_img)
				val_SwinT_loss = Loss_SwinT(val_SwinT_y, val_label)
				val_ViG_loss = Loss_ViG(val_ViG_y, val_label)
				val_Fusion_loss = Loss_Fusion(val_Fusion_y, val_label)
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number),'\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  )

	fusion_net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.to(gpu_device)
		test_label = test_label.to(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y,_,_ = fusion_net(test_img)
			test_SwinT_loss = Loss_SwinT(test_SwinT_y, test_label)
			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))
			test_VMamba_loss = Loss_ViG(test_ViG_y, test_label)
			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))
			test_Fusion_loss = Loss_Fusion(test_Fusion_y, test_label)
			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('########################## testing results #########################''\n',)
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
			'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
			' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
			'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
			' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)),'\n',
		  )
	g = fusion_net.state_dict()
	torch.save(g, weight_path)
	return test_Fusion_acc


def train_bw_EMCO(number=None, fusion_net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
					 fusion_weight=None, probability_weight=None, entropy_weight=None):
	fusion_net = fusion_net.to(gpu_device)
	Loss_SwinT = nn.CrossEntropyLoss()
	Loss_ViG = nn.CrossEntropyLoss()
	Loss_Fusion = nn.CrossEntropyLoss()

	for i in range(epoch):
		start_time = time.time()
		optim = torch.optim.AdamW(fusion_net.parameters(), cnn_lr_schedule(i))
		fusion_net.train()
		for img_data, img_label in train_loader:
			img_data = img_data.to(gpu_device)
			img_label = img_label.to(gpu_device)

			swint_y, vig_y, fusion_y, swin_feature, vig_feature = fusion_net(img_data)
			swin_feature = swin_feature.squeeze(-1)
			_, _, v1 = torch.linalg.svd(swin_feature)
			swin_feature = swin_feature @ v1[:10, :].mT
			_, _, v2 = torch.linalg.svd(vig_feature)
			vig_feature = vig_feature @ v2[:10, :].mT

			model1 = LinearRegression()
			model1.fit(swin_feature.detach().cpu().numpy(), img_label.detach().cpu().numpy())
			R1_squared = model1.score(swin_feature.detach().cpu().numpy(), img_label.detach().cpu().numpy())

			model2 = LinearRegression()
			model2.fit(vig_feature.detach().cpu().numpy(), img_label.detach().cpu().numpy())
			R2_squared = model2.score(vig_feature.detach().cpu().numpy(), img_label.detach().cpu().numpy())

			swin_weight = torch.tensor(R1_squared / (R1_squared + R2_squared))
			vig_weight = torch.tensor(R2_squared / (R1_squared + R2_squared))
			#print(swin_weight, vig_weight)
			loss_SwinT = Loss_SwinT(swint_y, img_label)
			loss_ViG = Loss_ViG(vig_y, img_label)
			loss_Fusion = Loss_Fusion(fusion_y, img_label)
			a = F.softmax(swint_y, dim=1)
			loss_entropy = torch.mean(-(F.softmax(swint_y, dim=1) * torch.log(F.softmax(vig_y, dim=1))))
			loss_pro = torch.mean(torch.abs(F.softmax(swint_y, dim=1) - F.softmax(vig_y, dim=1)))
			loss = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vig_weight * loss_ViG + probability_weight * loss_pro + entropy_weight * loss_entropy

			loss.backward()
			optim.step()
			optim.zero_grad()

		fusion_net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []

		for train_img, train_label in train_loader:
			train_img = train_img.to(gpu_device)
			train_label = train_label.to(gpu_device)
			with torch.no_grad():
				#print(train_label)
				train_SwinT_y, train_ViG_y, train_Fusion_y,_,_ = fusion_net(train_img)
				#print(train_SwinT_y)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.to(gpu_device)
			val_label = val_label.to(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y,_,_ = fusion_net(val_img)
				val_SwinT_loss = Loss_SwinT(val_SwinT_y, val_label)
				val_ViG_loss = Loss_ViG(val_ViG_y, val_label)
				val_Fusion_loss = Loss_Fusion(val_Fusion_y, val_label)
				val_loss_entropy = torch.mean(-(F.softmax(val_ViG_y, dim=1) * torch.log(F.softmax(val_SwinT_y, dim=1))))
				val_Pro_loss = torch.mean(torch.abs(F.softmax(val_ViG_y, dim=1) - F.softmax(val_SwinT_y, dim=1)))
				val_loss_sum = fusion_weight * val_Fusion_loss + probability_weight * val_Pro_loss + entropy_weight * val_loss_entropy

				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number),'\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',

			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()),
			  'ent_loss:{:.4}'.format(val_loss_entropy.detach().cpu().numpy()),
			  'pro_loss:{:.4}'.format(val_Pro_loss.detach().cpu().numpy())
			  )

	fusion_net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.to(gpu_device)
		test_label = test_label.to(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y,_,_ = fusion_net(test_img)
			test_SwinT_loss = Loss_SwinT(test_SwinT_y, test_label)
			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))
			test_VMamba_loss = Loss_ViG(test_ViG_y, test_label)
			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))
			test_Fusion_loss = Loss_Fusion(test_Fusion_y, test_label)
			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('########################## testing results #########################''\n',)
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
			'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
			' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
			'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
			' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)),'\n',
		  )
	g = fusion_net.state_dict()
	torch.save(g, weight_path)
	return test_Fusion_acc

def train_moo(number=None, net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
			 fusion_weight=None, swin_weight=None, vig_weight=None):
	net = net.to(gpu_device)
	SwinT_loss = nn.CrossEntropyLoss()
	ViG_loss = nn.CrossEntropyLoss()
	Fusion_loss = nn.CrossEntropyLoss()
	for i in range(epoch):
		start_time = time.time()
		para = [{'params': net.stem.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.prediction.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.vig_out.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_0.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_1.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_2.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_3.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_4.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_5.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_6.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_7.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_8.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_9.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_10.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_11.parameters(), 'lr': cnn_lr_schedule(i)},

				{'params': net.patch_embed.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.norm.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.head.parameters(), 'lr': vit_lr_schedule(i)}]
		optim = torch.optim.AdamW(para)
		net.train()

		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label = img_label.cuda(gpu_device)

			swint_y, vig_y, fusion_y = net(img_data)

			loss_SwinT = SwinT_loss(swint_y, img_label)
			loss_ViG = ViG_loss(vig_y, img_label)
			loss_Fusion = Fusion_loss(fusion_y, img_label)
			loss_sum = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vig_weight * loss_ViG

			loss_sum.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_SwinT_y, train_ViG_y, train_Fusion_y = net(train_img)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y = net(val_img)
				val_SwinT_loss = SwinT_loss(val_SwinT_y, val_label)
				val_ViG_loss = ViG_loss(val_ViG_y, val_label)
				val_Fusion_loss = Fusion_loss(val_Fusion_y, val_label)
				val_loss_sum = fusion_weight * val_Fusion_loss + swin_weight * val_SwinT_loss + vig_weight * val_ViG_loss
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number), '\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()))

	net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y = net(test_img)

			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))

			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))

			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
		' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
		' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
		'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
		' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
		' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
		'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
		' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
		' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = net.state_dict()
	torch.save(g, weight_path)

	return test_Fusion_acc


def train_moo_fdaf(number=None, net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
			 fusion_weight=None, swin_weight=None, vig_weight=None):
	net = net.to(gpu_device)
	SwinT_loss = nn.CrossEntropyLoss()
	ViG_loss = nn.CrossEntropyLoss()
	Fusion_loss = nn.CrossEntropyLoss()
	for i in range(epoch):
		start_time = time.time()
		para = [{'params': net.stem.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.prediction.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.vig_out.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_0.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_1.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_2.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_3.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_4.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_5.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_6.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_7.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_8.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_9.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_10.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_11.parameters(), 'lr': cnn_lr_schedule_2(i)},

				{'params': net.patch_embed.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.norm.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.head.parameters(), 'lr': vit_lr_schedule(i)},

				{'params': net.fusion_block_dense_1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_block_dense_2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_layer.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				]
		optim = torch.optim.AdamW(para)
		net.train()

		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label = img_label.cuda(gpu_device)

			swint_y, vig_y, fusion_y = net(img_data)

			loss_SwinT = SwinT_loss(swint_y, img_label)
			loss_ViG = ViG_loss(vig_y, img_label)
			loss_Fusion = Fusion_loss(fusion_y, img_label)
			loss_sum = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vig_weight * loss_ViG

			loss_sum.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_SwinT_y, train_ViG_y, train_Fusion_y = net(train_img)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y = net(val_img)
				val_SwinT_loss = SwinT_loss(val_SwinT_y, val_label)
				val_ViG_loss = ViG_loss(val_ViG_y, val_label)
				val_Fusion_loss = Fusion_loss(val_Fusion_y, val_label)
				val_loss_sum = fusion_weight * val_Fusion_loss + swin_weight * val_SwinT_loss + vig_weight * val_ViG_loss
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number), '\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()))

	net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y = net(test_img)

			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))

			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))

			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
		' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
		' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
		'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
		' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
		' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
		'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
		' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
		' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = net.state_dict()
	torch.save(g, weight_path)

	return test_Fusion_acc


def train_moo_aamff(number=None, net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
			 fusion_weight=None, swin_weight=None, vig_weight=None):
	net = net.to(gpu_device)
	SwinT_loss = nn.CrossEntropyLoss()
	ViG_loss = nn.CrossEntropyLoss()
	Fusion_loss = nn.CrossEntropyLoss()
	for i in range(epoch):
		start_time = time.time()
		para = [{'params': net.stem.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.prediction.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.vig_out.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_0.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_1.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_2.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_3.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_4.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_5.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_6.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_7.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_8.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_9.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_10.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_11.parameters(), 'lr': cnn_lr_schedule(i)},

				{'params': net.patch_embed.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.norm.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.head.parameters(), 'lr': vit_lr_schedule(i)},

				{'params': net.swin_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.swin_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				]
		optim = torch.optim.AdamW(para)
		net.train()

		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label = img_label.cuda(gpu_device)

			swint_y, vig_y, fusion_y = net(img_data)

			loss_SwinT = SwinT_loss(swint_y, img_label)
			loss_ViG = ViG_loss(vig_y, img_label)
			loss_Fusion = Fusion_loss(fusion_y, img_label)
			loss_sum = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vig_weight * loss_ViG

			loss_sum.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_SwinT_y, train_ViG_y, train_Fusion_y = net(train_img)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y = net(val_img)
				val_SwinT_loss = SwinT_loss(val_SwinT_y, val_label)
				val_ViG_loss = ViG_loss(val_ViG_y, val_label)
				val_Fusion_loss = Fusion_loss(val_Fusion_y, val_label)
				val_loss_sum = fusion_weight * val_Fusion_loss + swin_weight * val_SwinT_loss + vig_weight * val_ViG_loss
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number), '\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()))

	net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y = net(test_img)

			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))

			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))

			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
		' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
		' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
		'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
		' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
		' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
		'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
		' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
		' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = net.state_dict()
	torch.save(g, weight_path)

	return test_Fusion_acc


def train_moo_dis(number=None, net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
			 fusion_weight=None, swin_weight=None, vmamba_weight=None, dis_weight=None):
	net = net.to(gpu_device)
	SwinT_loss = nn.CrossEntropyLoss()
	ViG_loss = nn.CrossEntropyLoss()
	Fusion_loss = nn.CrossEntropyLoss()
	Dis_loss = adv.AdversarialLoss().to(gpu_device)  # lmmd.LMMDLoss()#mmd.MMD_loss()#nn.TripletMarginLoss()#daan.DAANLoss(num_class=3)#adv.AdversarialLoss()#coral.CORAL()#
	LMMD_Dis = lmmd.LMMDLoss().to(gpu_device)
	for i in range(epoch):
		start_time = time.time()
		para = [{'params': net.stem.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.prediction.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.vig_out.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_0.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_1.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_2.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_3.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_4.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_5.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_6.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_7.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_8.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_9.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_10.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.block_11.parameters(), 'lr': cnn_lr_schedule_2(i)},

				{'params': net.patch_embed.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.norm.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.head.parameters(), 'lr': vit_lr_schedule(i)},

				{'params': net.fusion_block_dense_1.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.fusion_block_dense_2.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': net.fusion_layer.parameters(), 'lr': cnn_lr_schedule_2(i)},
				{'params': Dis_loss.parameters(), 'lr': cnn_lr_schedule_2(i)},
				]
		optim = torch.optim.AdamW(para)
		net.train()

		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label = img_label.cuda(gpu_device)

			swint_y, vig_y, fusion_y, swint_y_fusion, vig_y_fusion = net(img_data)
			swint_y_fusion = swint_y_fusion.squeeze(-1)
			_, _, v1 = torch.linalg.svd(swint_y_fusion)
			swint_y_fusion = swint_y_fusion @ v1[:30, :].mT
			_, _, v2 = torch.linalg.svd(vig_y_fusion)
			vig_y_fusion = vig_y_fusion @ v2[:30, :].mT

			loss_SwinT = SwinT_loss(swint_y, img_label)
			loss_ViG = ViG_loss(vig_y, img_label)
			loss_Fusion = Fusion_loss(fusion_y, img_label)

			loss_Dis = Dis_loss(swint_y_fusion, vig_y_fusion)
			#loss_LMMD = LMMD_Dis(vig_y_fusion, swint_y_fusion, vig_y, swint_y, gpu_device)[0]
			loss_sum = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vmamba_weight * loss_ViG + dis_weight * loss_Dis

			loss_sum.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_SwinT_y, train_ViG_y, train_Fusion_y,_,_ = net(train_img)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y, val_SwinT_fusion, val_ViG_fusion = net(val_img)
				val_SwinT_fusion = val_SwinT_fusion.squeeze(-1)
				val_SwinT_loss = SwinT_loss(val_SwinT_y, val_label)
				val_ViG_loss = ViG_loss(val_ViG_y, val_label)
				val_Fusion_loss = Fusion_loss(val_Fusion_y, val_label)


				_, _, vs = torch.linalg.svd(val_SwinT_fusion)
				val_SwinT_fusion = val_SwinT_fusion @ vs[:30, :].T
				_, _, vv = torch.linalg.svd(val_ViG_fusion)
				val_ViG_fusion = val_ViG_fusion @ vv[:30, :].T

				val_Dis_loss = Dis_loss(val_SwinT_fusion, val_ViG_fusion)
				val_loss_sum = fusion_weight * val_Fusion_loss + swin_weight * val_SwinT_loss + vmamba_weight * val_ViG_loss + dis_weight * val_Dis_loss
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number), '\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_dis_loss:{:.4}'.format(val_Dis_loss.detach().cpu().numpy()),
			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()))

	net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y,_,_ = net(test_img)

			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))

			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))

			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
		' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
		' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
		'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
		' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
		' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
		'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
		' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
		' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = net.state_dict()
	torch.save(g, weight_path)

	return test_Fusion_acc

def train_moo_dis_adv(number=None, net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
			 fusion_weight=None, swin_weight=None, vmamba_weight=None, dis_weight=None):
	net = net.to(gpu_device)
	SwinT_loss = nn.CrossEntropyLoss()
	ViG_loss = nn.CrossEntropyLoss()
	Fusion_loss = nn.CrossEntropyLoss()
	Dis_loss = adv.AdversarialLoss().to(gpu_device)  # lmmd.LMMDLoss()#mmd.MMD_loss()#nn.TripletMarginLoss()#daan.DAANLoss(num_class=3)#adv.AdversarialLoss()#coral.CORAL()#
	#LMMD_Dis = lmmd.LMMDLoss().to(gpu_device)
	for i in range(epoch):
		start_time = time.time()
		para = [{'params': net.stem.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.prediction.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.vig_out.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_0.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_1.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_2.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_3.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_4.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_5.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_6.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_7.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_8.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_9.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_10.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_11.parameters(), 'lr': cnn_lr_schedule(i)},

				{'params': net.patch_embed.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.norm.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.head.parameters(), 'lr': vit_lr_schedule(i)},

				{'params': net.swin_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.swin_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': Dis_loss.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				]
		optim = torch.optim.AdamW(para)
		net.train()

		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label = img_label.cuda(gpu_device)

			swint_y, vig_y, fusion_y, swint_y_fusion, vig_y_fusion = net(img_data)
			swint_y_fusion = swint_y_fusion.squeeze(-1)
			_, _, v1 = torch.linalg.svd(swint_y_fusion)
			swint_y_fusion = swint_y_fusion @ v1[:50, :].mT
			_, _, v2 = torch.linalg.svd(vig_y_fusion)
			vig_y_fusion = vig_y_fusion @ v2[:50, :].mT

			loss_SwinT = SwinT_loss(swint_y, img_label)
			loss_ViG = ViG_loss(vig_y, img_label)
			loss_Fusion = Fusion_loss(fusion_y, img_label)

			loss_Dis = Dis_loss(swint_y_fusion, vig_y_fusion)
			#loss_LMMD = LMMD_Dis(vig_y_fusion, swint_y_fusion, vig_y, swint_y, gpu_device)[0]
			loss_sum = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vmamba_weight * loss_ViG + dis_weight * loss_Dis

			loss_sum.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_SwinT_y, train_ViG_y, train_Fusion_y,_,_ = net(train_img)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y, val_SwinT_fusion, val_ViG_fusion = net(val_img)
				val_SwinT_fusion = val_SwinT_fusion.squeeze(-1)
				val_SwinT_loss = SwinT_loss(val_SwinT_y, val_label)
				val_ViG_loss = ViG_loss(val_ViG_y, val_label)
				val_Fusion_loss = Fusion_loss(val_Fusion_y, val_label)


				_, _, vs = torch.linalg.svd(val_SwinT_fusion)
				val_SwinT_fusion = val_SwinT_fusion @ vs[:50, :].T
				_, _, vv = torch.linalg.svd(val_ViG_fusion)
				val_ViG_fusion = val_ViG_fusion @ vv[:50, :].T

				val_Dis_loss = Dis_loss(val_SwinT_fusion, val_ViG_fusion)
				#val_LMMD_loss = LMMD_Dis(val_ViG_fusion, val_SwinT_fusion, val_ViG_y, val_SwinT_y, gpu_device)[0]
				val_loss_sum = fusion_weight * val_Fusion_loss + swin_weight * val_SwinT_loss + vmamba_weight * val_ViG_loss + dis_weight * val_Dis_loss
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number), '\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_dis_loss:{:.4}'.format(val_Dis_loss.detach().cpu().numpy()),
			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()))

	net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y,_,_ = net(test_img)

			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))

			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))

			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
		' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
		' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
		'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
		' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
		' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
		'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
		' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
		' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = net.state_dict()
	torch.save(g, weight_path)

	return test_Fusion_acc


def train_moo_dis_mmd(number=None, net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
			 fusion_weight=None, swin_weight=None, vmamba_weight=None, dis_weight=None):
	net = net.to(gpu_device)
	SwinT_loss = nn.CrossEntropyLoss()
	ViG_loss = nn.CrossEntropyLoss()
	Fusion_loss = nn.CrossEntropyLoss()
	Dis_loss = mmd.MMD_loss().to(gpu_device)  # lmmd.LMMDLoss()#mmd.MMD_loss()#nn.TripletMarginLoss()#daan.DAANLoss(num_class=3)#adv.AdversarialLoss()#coral.CORAL()#
	#LMMD_Dis = lmmd.LMMDLoss().to(gpu_device)
	for i in range(epoch):
		start_time = time.time()
		para = [{'params': net.stem.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.prediction.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.vig_out.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_0.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_1.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_2.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_3.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_4.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_5.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_6.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_7.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_8.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_9.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_10.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_11.parameters(), 'lr': cnn_lr_schedule(i)},

				{'params': net.patch_embed.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.norm.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.head.parameters(), 'lr': vit_lr_schedule(i)},

				{'params': net.swin_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.swin_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': Dis_loss.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				]
		optim = torch.optim.AdamW(para)
		net.train()

		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label = img_label.cuda(gpu_device)

			swint_y, vig_y, fusion_y, swint_y_fusion, vig_y_fusion = net(img_data)
			swint_y_fusion = swint_y_fusion.squeeze(-1)
			_, _, v1 = torch.linalg.svd(swint_y_fusion)
			swint_y_fusion = swint_y_fusion @ v1[:50, :].mT
			_, _, v2 = torch.linalg.svd(vig_y_fusion)
			vig_y_fusion = vig_y_fusion @ v2[:50, :].mT

			loss_SwinT = SwinT_loss(swint_y, img_label)
			loss_ViG = ViG_loss(vig_y, img_label)
			loss_Fusion = Fusion_loss(fusion_y, img_label)

			loss_Dis = Dis_loss(swint_y_fusion, vig_y_fusion)
			#loss_LMMD = LMMD_Dis(vig_y_fusion, swint_y_fusion, vig_y, swint_y, gpu_device)[0]
			loss_sum = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vmamba_weight * loss_ViG + dis_weight * loss_Dis

			loss_sum.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_SwinT_y, train_ViG_y, train_Fusion_y,_,_ = net(train_img)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y, val_SwinT_fusion, val_ViG_fusion = net(val_img)
				val_SwinT_fusion = val_SwinT_fusion.squeeze(-1)
				val_SwinT_loss = SwinT_loss(val_SwinT_y, val_label)
				val_ViG_loss = ViG_loss(val_ViG_y, val_label)
				val_Fusion_loss = Fusion_loss(val_Fusion_y, val_label)


				_, _, vs = torch.linalg.svd(val_SwinT_fusion)
				val_SwinT_fusion = val_SwinT_fusion @ vs[:50, :].T
				_, _, vv = torch.linalg.svd(val_ViG_fusion)
				val_ViG_fusion = val_ViG_fusion @ vv[:50, :].T

				val_Dis_loss = Dis_loss(val_SwinT_fusion, val_ViG_fusion)
				#val_LMMD_loss = LMMD_Dis(val_ViG_fusion, val_SwinT_fusion, val_ViG_y, val_SwinT_y, gpu_device)[0]
				val_loss_sum = fusion_weight * val_Fusion_loss + swin_weight * val_SwinT_loss + vmamba_weight * val_ViG_loss + dis_weight * val_Dis_loss
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number), '\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_dis_loss:{:.4}'.format(val_Dis_loss.detach().cpu().numpy()),
			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()))

	net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y,_,_ = net(test_img)

			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))

			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))

			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
		' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
		' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
		'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
		' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
		' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
		'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
		' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
		' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = net.state_dict()
	torch.save(g, weight_path)

	return test_Fusion_acc


def train_moo_dis_lmmd(number=None, net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
			 fusion_weight=None, swin_weight=None, vmamba_weight=None, dis_weight=None):
	net = net.to(gpu_device)
	SwinT_loss = nn.CrossEntropyLoss()
	ViG_loss = nn.CrossEntropyLoss()
	Fusion_loss = nn.CrossEntropyLoss()
	LMMD_Dis = lmmd.LMMDLoss().to(gpu_device)  # lmmd.LMMDLoss()#mmd.MMD_loss()#nn.TripletMarginLoss()#daan.DAANLoss(num_class=3)#adv.AdversarialLoss()#coral.CORAL()#
	#LMMD_Dis =
	for i in range(epoch):
		start_time = time.time()
		para = [{'params': net.stem.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.prediction.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.vig_out.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_0.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_1.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_2.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_3.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_4.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_5.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_6.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_7.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_8.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_9.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_10.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_11.parameters(), 'lr': cnn_lr_schedule(i)},

				{'params': net.patch_embed.parameters(), 'lr': vit_lr_greece_larynx(i)},
				{'params': net.layers_0.parameters(), 'lr': vit_lr_greece_larynx(i)},
				{'params': net.layers_1.parameters(), 'lr': vit_lr_greece_larynx(i)},
				{'params': net.layers_2.parameters(), 'lr': vit_lr_greece_larynx(i)},
				{'params': net.layers_3.parameters(), 'lr': vit_lr_greece_larynx(i)},
				{'params': net.norm.parameters(), 'lr': vit_lr_greece_larynx(i)},
				{'params': net.head.parameters(), 'lr': vit_lr_greece_larynx(i)},

				{'params': net.swin_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.swin_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				#{'params': Dis_loss.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				]
		optim = torch.optim.AdamW(para)
		net.train()

		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label = img_label.cuda(gpu_device)

			swint_y, vig_y, fusion_y, swint_y_fusion, vig_y_fusion = net(img_data)
			swint_y_fusion = swint_y_fusion.squeeze(-1)
			_, _, v1 = torch.linalg.svd(swint_y_fusion)
			swint_y_fusion = swint_y_fusion @ v1[:50, :].mT
			_, _, v2 = torch.linalg.svd(vig_y_fusion)
			vig_y_fusion = vig_y_fusion @ v2[:50, :].mT

			loss_SwinT = SwinT_loss(swint_y, img_label)
			loss_ViG = ViG_loss(vig_y, img_label)
			loss_Fusion = Fusion_loss(fusion_y, img_label)

			loss_Dis = LMMD_Dis(vig_y_fusion, swint_y_fusion, vig_y, swint_y, gpu_device).item()
			#loss_LMMD = LMMD_Dis(vig_y_fusion, swint_y_fusion, vig_y, swint_y, gpu_device)[0]
			loss_sum = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vmamba_weight * loss_ViG + dis_weight * loss_Dis

			loss_sum.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_SwinT_y, train_ViG_y, train_Fusion_y,_,_ = net(train_img)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y, val_SwinT_fusion, val_ViG_fusion = net(val_img)
				val_SwinT_fusion = val_SwinT_fusion.squeeze(-1)
				val_SwinT_loss = SwinT_loss(val_SwinT_y, val_label)
				val_ViG_loss = ViG_loss(val_ViG_y, val_label)
				val_Fusion_loss = Fusion_loss(val_Fusion_y, val_label)


				_, _, vs = torch.linalg.svd(val_SwinT_fusion)
				val_SwinT_fusion = val_SwinT_fusion @ vs[:50, :].T
				_, _, vv = torch.linalg.svd(val_ViG_fusion)
				val_ViG_fusion = val_ViG_fusion @ vv[:50, :].T

				val_Dis_loss = LMMD_Dis(val_ViG_fusion, val_SwinT_fusion, val_ViG_y, val_SwinT_y, gpu_device).item()
				#val_LMMD_loss = LMMD_Dis(val_ViG_fusion, val_SwinT_fusion, val_ViG_y, val_SwinT_y, gpu_device)[0]
				val_loss_sum = fusion_weight * val_Fusion_loss + swin_weight * val_SwinT_loss + vmamba_weight * val_ViG_loss + dis_weight * val_Dis_loss
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number), '\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_dis_loss:{:.4}'.format(val_Dis_loss),
			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()))

	net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y,_,_ = net(test_img)

			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))

			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))

			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
		' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
		' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
		'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
		' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
		' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
		'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
		' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
		' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = net.state_dict()
	torch.save(g, weight_path)

	return test_Fusion_acc


def train_moo_dis_euclidean(number=None, net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
			 fusion_weight=None, swin_weight=None, vmamba_weight=None, dis_weight=None):
	net = net.to(gpu_device)
	SwinT_loss = nn.CrossEntropyLoss()
	ViG_loss = nn.CrossEntropyLoss()
	Fusion_loss = nn.CrossEntropyLoss()
	Dis_loss = coral.CORAL()
	for i in range(epoch):
		start_time = time.time()
		para = [{'params': net.stem.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.prediction.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.vig_out.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_0.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_1.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_2.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_3.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_4.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_5.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_6.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_7.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_8.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_9.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_10.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_11.parameters(), 'lr': cnn_lr_schedule(i)},

				{'params': net.patch_embed.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.norm.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.head.parameters(), 'lr': vit_lr_schedule(i)},

				{'params': net.swin_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.swin_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.vig_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc1.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				{'params': net.fusion_fc2.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				#{'params': Dis_loss.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				]
		optim = torch.optim.AdamW(para)
		net.train()

		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label = img_label.cuda(gpu_device)

			swint_y, vig_y, fusion_y, swint_y_fusion, vig_y_fusion = net(img_data)
			swint_y_fusion = swint_y_fusion.squeeze(-1)
			_, _, v1 = torch.linalg.svd(swint_y_fusion)
			swint_y_fusion = swint_y_fusion @ v1[:50, :].mT
			_, _, v2 = torch.linalg.svd(vig_y_fusion)
			vig_y_fusion = vig_y_fusion @ v2[:50, :].mT

			loss_SwinT = SwinT_loss(swint_y, img_label)
			loss_ViG = ViG_loss(vig_y, img_label)
			loss_Fusion = Fusion_loss(fusion_y, img_label)

			loss_Dis = Dis_loss(swint_y_fusion, vig_y_fusion)#(torch.sum((swint_y_fusion - vig_y_fusion) ** 2) / (swint_y_fusion.shape[0] * vig_y_fusion.shape[1])) ** 0.5
			loss_sum = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vmamba_weight * loss_ViG + dis_weight * loss_Dis

			loss_sum.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_SwinT_y, train_ViG_y, train_Fusion_y,_,_ = net(train_img)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y, val_SwinT_fusion, val_ViG_fusion = net(val_img)
				val_SwinT_fusion = val_SwinT_fusion.squeeze(-1)
				val_SwinT_loss = SwinT_loss(val_SwinT_y, val_label)
				val_ViG_loss = ViG_loss(val_ViG_y, val_label)
				val_Fusion_loss = Fusion_loss(val_Fusion_y, val_label)


				_, _, vs = torch.linalg.svd(val_SwinT_fusion)
				val_SwinT_fusion = val_SwinT_fusion @ vs[:50, :].T
				_, _, vv = torch.linalg.svd(val_ViG_fusion)
				val_ViG_fusion = val_ViG_fusion @ vv[:50, :].T

				val_Dis_loss = Dis_loss(val_SwinT_fusion, val_ViG_fusion)#(torch.sum((val_SwinT_fusion - val_ViG_fusion) ** 2) / (val_SwinT_fusion.shape[0] * val_ViG_fusion.shape[1])) ** 0.5

				#val_LMMD_loss = LMMD_Dis(val_ViG_fusion, val_SwinT_fusion, val_ViG_y, val_SwinT_y, gpu_device)[0]
				val_loss_sum = fusion_weight * val_Fusion_loss + swin_weight * val_SwinT_loss + vmamba_weight * val_ViG_loss + dis_weight * val_Dis_loss
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number), '\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_dis_loss:{:.4}'.format(val_Dis_loss),
			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()))

	net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y,_,_ = net(test_img)

			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))

			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))

			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
		' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
		' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
		'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
		' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
		' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
		'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
		' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
		' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = net.state_dict()
	torch.save(g, weight_path)

	return test_Fusion_acc


def testing_funnction(test_model=None, train_loader=None, val_loader=None, test_loader=None,
                      gpu_device=0, out_mode=None, weight_path=None, log_path=None):
	"""Evaluate GAFT and persist classification, confusion-matrix, and AUC metrics."""
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
			images = images.cuda(gpu_device)
			labels = labels.cuda(gpu_device)
			with torch.no_grad():
				predictions = torch.argmax(_forward(images), dim=1)
			accuracies.append(accuracy_score(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy()))
		return float(np.mean(accuracies)) if accuracies else float('nan')

	test_model.eval()
	train_accuracy = _mean_accuracy(train_loader)
	val_accuracy = _mean_accuracy(val_loader)

	true_labels, predicted_labels = [], []
	test_accuracies = []
	for images, labels in test_loader:
		images = images.cuda(gpu_device)
		labels = labels.cuda(gpu_device)
		with torch.no_grad():
			logits = _forward(images)
			predictions = torch.argmax(logits, dim=1)
		labels_cpu = labels.detach().cpu().numpy()
		predictions_cpu = predictions.detach().cpu().numpy()
		test_accuracies.append(accuracy_score(labels_cpu, predictions_cpu))
		true_labels.append(labels_cpu)
		predicted_labels.append(predictions_cpu)

	y_true = np.concatenate(true_labels).astype(int)
	y_pred = np.concatenate(predicted_labels).astype(int)
	class_count = int(max(np.max(y_true), np.max(y_pred))) + 1
	labels = list(range(class_count))
	class_report = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
	matrix, confusion_metrics_report = confusion_matrix_metrics(y_true, y_pred, labels=labels)
	y_true_one_hot = one_hot(org_x=y_true, pre_dim=class_count)
	y_pred_one_hot = one_hot(org_x=y_pred, pre_dim=class_count)
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
		os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
		with open(log_path, 'a', encoding='utf-8') as log_file:
			log_file.write('########################## testing results #########################\n')
			log_file.write('train_acc:{:.4f}  val_acc:{:.4f}  test_acc:{:.4f}\n'.format(train_accuracy, val_accuracy, test_accuracy))
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


def testing_funnction_dis(test_model = None, train_loader=None, val_loader=None, test_loader=None, gpu_device=0,
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
				_, _, train_pre_y,_,_ = test_model(train_img)
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
				_, _, val_pre_y ,_,_= test_model(val_img)
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
				_, _, test_pre_y,_,_ = test_model(test_img)

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

def cnn_lr_schedule(epoch):
	if epoch < 50:
		lr = 1e-4
	elif epoch < 75:
		lr = 2e-5
	else:
		lr = 1e-6
	return lr


def cnn_lr_schedule_2(epoch):
	if epoch < 50:
		lr = 5e-4
	elif epoch < 75:
		lr = 1e-4
	else:
		lr = 5e-5
	return lr

def cnn_lr_schedule_3(epoch):
	if epoch < 50:
		lr = 5e-5
	elif epoch < 75:
		lr = 1e-4
	else:
		lr = 5e-4
	return lr

def vit_lr_schedule(epoch):
	if epoch < 50:
		lr = 1e-5
	elif epoch < 75:
		lr = 5e-6
	else:
		lr = 1e-6
	return lr

def vit_lr_greece_larynx(epoch):
	if epoch < 50:
		lr = 3e-5
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


def one_hot(org_x = None, pre_dim = 3):
	one_x = np.zeros((org_x.shape[0], pre_dim))
	for i in range(org_x.shape[0]):
		one_x[i, int(org_x[i])] = 1
	return one_x


class CapturePrint:
	def __init__(self):
		self.contents = ""

	def write(self, text):
		self.contents += text

	def getvalue(self):
		return self.contents


class PrintCapture:
	def __init__(self, capture):
		self.capture = capture

	def write(self, text):
		self.capture.write(text)
		sys.__stdout__.write(text)

	def getvalue(self):
		return self.capture.getvalue()


def train_moo_cont(number=None, net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None,
			 fusion_weight=None, swin_weight=None, vig_weight=None):
	net = net.to(gpu_device)
	SwinT_loss = nn.CrossEntropyLoss()
	ViG_loss = nn.CrossEntropyLoss()
	Fusion_loss = nn.CrossEntropyLoss()
	for i in range(epoch):
		start_time = time.time()
		para = [{'params': net.stem.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.prediction.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.vig_out.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_0.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_1.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_2.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_3.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_4.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_5.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_6.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_7.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_8.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_9.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_10.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': net.block_11.parameters(), 'lr': cnn_lr_schedule(i)},

				{'params': net.patch_embed.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.layers_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.norm.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': net.head.parameters(), 'lr': vit_lr_schedule(i)},

				{'params': net.fusion_fc.parameters(), 'lr': vit_lr_for_breast_schedule(i)},
				]
		optim = torch.optim.AdamW(para)
		net.train()

		for img_data, img_label in train_loader:
			img_data = img_data.cuda(gpu_device)
			img_label = img_label.cuda(gpu_device)

			swint_y, vig_y, fusion_y = net(img_data)

			loss_SwinT = SwinT_loss(swint_y, img_label)
			loss_ViG = ViG_loss(vig_y, img_label)
			loss_Fusion = Fusion_loss(fusion_y, img_label)
			loss_sum = fusion_weight * loss_Fusion + swin_weight * loss_SwinT + vig_weight * loss_ViG

			loss_sum.backward()
			optim.step()
			optim.zero_grad()

		net.eval()
		train_SwinT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []
		for train_img, train_label in train_loader:
			train_img = train_img.cuda(gpu_device)
			train_label = train_label.cuda(gpu_device)
			with torch.no_grad():
				train_SwinT_y, train_ViG_y, train_Fusion_y = net(train_img)
				pre_train_SwinT = torch.argmax(train_SwinT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_SwinT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_SwinT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_SwinT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.cuda(gpu_device)
			val_label = val_label.cuda(gpu_device)
			with torch.no_grad():
				val_SwinT_y, val_ViG_y, val_Fusion_y = net(val_img)
				val_SwinT_loss = SwinT_loss(val_SwinT_y, val_label)
				val_ViG_loss = ViG_loss(val_ViG_y, val_label)
				val_Fusion_loss = Fusion_loss(val_Fusion_y, val_label)
				val_loss_sum = fusion_weight * val_Fusion_loss + swin_weight * val_SwinT_loss + vig_weight * val_ViG_loss
				pre_val_SwinT = torch.argmax(val_SwinT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_SwinT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_SwinT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number), '\n',
			  'epoch ' + str(i + 1), ' train_SwinT_loss:{:.4}'.format(loss_SwinT.detach().cpu().numpy()),
			  ' train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
			  ' val_SwinT_loss:{:.4}'.format(val_SwinT_loss.detach().cpu().numpy()),
			  ' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
			  ' Time:{:.3}'.format(end_time - start_time), '\n',
			  '________train_ViG_loss:{:.4}'.format(loss_ViG.detach().cpu().numpy()),
			  ' train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			  ' val_ViG_loss:{:.4}'.format(val_ViG_loss.detach().cpu().numpy()),
			  ' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)), '\n',
			  '________train_Fusion_loss:{:.4}'.format(loss_Fusion.detach().cpu().numpy()),
			  ' train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			  '   val_Fusion_loss:{:.4}'.format(val_Fusion_loss.detach().cpu().numpy()),
			  ' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)), '\n',
			  'val_loss:{:.4}'.format(val_loss_sum.detach().cpu().numpy()))

	net.eval()
	test_SwinT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.cuda(gpu_device)
		test_label = test_label.cuda(gpu_device)
		with torch.no_grad():
			test_SwinT_y, test_ViG_y, test_Fusion_y = net(test_img)

			pre_test_SwinT = torch.argmax(test_SwinT_y, dim=1)
			test_SwinT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_SwinT.detach().cpu().numpy()))

			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))

			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('train_SwinT_acc:{:.4}'.format(np.mean(train_SwinT_acc)),
		' val_SwinT_acc:{:.4}'.format(np.mean(val_SwinT_acc)),
		' test_SwinT_acc:{:.4}'.format(np.mean(test_SwinT_acc)), '\n',
		'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
		' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
		' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
		'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
		' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
		' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = net.state_dict()
	torch.save(g, weight_path)

	return test_Fusion_acc
