import torch
import numpy as np
import random
import sys
import time
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, roc_curve, accuracy_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score
import torch.nn.functional as F
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


def testing_funnction(test_model = None, train_loader=None, val_loader=None, test_loader=None, gpu_device=0,
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
			elif out_mode == 'five':
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
				_, _, val_pre_y = test_model(val_img)
			elif out_mode == 'five':
				_, _, val_pre_y,_,_ = test_model(val_img)
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
			elif out_mode == 'five':
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

	# ##########################################Confusion Matrix###############################################
	# y_gt = []
	# y_pred = []
	# for imgs, labels in test_loader:
	# 	imgs = imgs.cuda(gpu_device)
	# 	labels = labels.cuda(gpu_device)
	# 	if out_mode == 'single':
	# 		labels_pd = test_model(imgs)
	# 	elif out_mode == 'triplet':
	# 		_, _, labels_pd = test_model(imgs)
	# 	elif out_mode == 'five':
	# 		_, _, labels_pd, _, _ = test_model(imgs)
	# 	predict_np = np.argmax(labels_pd.cpu().detach().numpy(), axis=-1)  # array([0,5,1,6,3,...],dtype=int64)
	# 	labels_np = labels.cpu().detach().numpy()  # array([0,5,0,6,2,...],dtype=int64)
	#
	# 	del labels_pd
	# 	labels_pd = None
	# 	torch.cuda.empty_cache()
	#
	# 	y_pred.append(predict_np)
	# 	y_gt.append(labels_np)
	#
	# result_y_pred = []
	# for sublist in y_pred:
	# 	result_y_pred.extend(sublist)
	# result_y_gt = []
	# for sublist in y_gt:
	# 	result_y_gt.extend(sublist)
	#
	# draw_confusion_matrix(label_true=np.array(result_y_gt),  # y_gt=[0,5,1,6,3,...]
	# 					  label_pred=np.array(result_y_pred),  # y_pred=[0,5,1,6,3,...]
	# 					  label_name=["Grade I","Grade II","Grade III"],
	# 					  title="Train Data",
	# 					  pdf_save_path="Train Data",
	# 					  dpi=300)
	# ##########################################MCC################################################################
	# mcc = matthews_corrcoef(result_y_gt, result_y_pred)
	# print("MCC:", mcc)
	# ##########################################kappa###############################################################
	# kappa = cohen_kappa_score(np.array(result_y_gt).reshape(-1, 1), np.array(result_y_pred).reshape(-1, 1))
	# print("Kappa:", kappa)
	#
	# ##########################################CSI###############################################################
	# def calculate_csi(true_positives, false_positives, false_negatives):
	# 	return true_positives / (true_positives + false_positives + false_negatives)
	#
	# result_y_pred = np.array(result_y_pred)
	# result_y_gt = np.array(result_y_gt)
	# true_positives = np.sum((result_y_pred == result_y_gt) & (result_y_pred > 0))
	# false_positives = np.sum((result_y_pred != result_y_gt) & (result_y_pred > 0))
	# false_negatives = np.sum((result_y_pred != result_y_gt) & (result_y_gt > 0))
	#
	# csi = calculate_csi(true_positives, false_positives, false_negatives)
	# print(f"Critical Success Index (CSI): {csi}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):

	""""@param label_true: 真实标签，比如[0,1,2,7,4,5,...]
	@param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
	@param label_name: 标签名字，比如['cat','dog','flower',...]
	@param title: 图标题
	@param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
	@param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
	@return:

	example：
			draw_confusion_matrix(label_true=y_gt,
						  label_pred=y_pred,
						  label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
						  title="Confusion Matrix on Fer2013",
						  pdf_save_path="Confusion_Matrix_on_Fer2013.png",
						  dpi=300)"""


	cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize=None)
	plt.rcParams['font.serif'] = ['Times New Roman']
	plt.imshow(cm, cmap='Blues')
	plt.grid(False)
	plt.title(title,fontsize=18)
	plt.xlabel("Predict label",fontsize=14)
	plt.ylabel("Truth label",fontsize=14)
	plt.yticks(range(label_name.__len__()), label_name,fontsize=12)
	plt.xticks(range(label_name.__len__()), label_name, rotation=15,fontsize=12)

	plt.tight_layout()

	plt.colorbar()

	for i in range(label_name.__len__()):
		for j in range(label_name.__len__()):
			color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
			value = float(format('%.3f' % cm[j, i]))
			plt.text(i, j, int(value), verticalalignment='center', horizontalalignment='center', color=color, fontsize=14)

	# plt.show()
	if not pdf_save_path is None:
		plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

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