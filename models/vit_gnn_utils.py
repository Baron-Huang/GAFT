import torch
import numpy as np
import random
import sys
import time
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, roc_curve, accuracy_score, roc_auc_score
from loss_funcs import adv, daan, coral, bnm, mmd, lmmd
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F

def train_baseline(number=None, fusion_net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, weight_path=None):
	fusion_net = fusion_net.to(gpu_device)
	Loss_ViT = nn.CrossEntropyLoss()
	Loss_ViG = nn.CrossEntropyLoss()
	Loss_Fusion = nn.CrossEntropyLoss()

	for i in range(epoch):
		start_time = time.time()
		optim = torch.optim.AdamW(fusion_net.parameters(), vit_lr_schedule(i))
		fusion_net.train()
		for img_data, img_label in train_loader:
			img_data = img_data.to(gpu_device)
			img_label = img_label.to(gpu_device)

			vit_y, vig_y, fusion_y = fusion_net(img_data)

			loss_ViT = Loss_ViT(vit_y, img_label)
			loss_ViG = Loss_ViG(vig_y, img_label)
			loss_Fusion = Loss_Fusion(fusion_y, img_label)

			loss = loss_Fusion

			loss.backward()
			optim.step()
			optim.zero_grad()

		fusion_net.eval()
		train_ViT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []

		for train_img, train_label in train_loader:
			train_img = train_img.to(gpu_device)
			train_label = train_label.to(gpu_device)
			with torch.no_grad():
				#print(train_label)
				train_ViT_y, train_ViG_y, train_Fusion_y = fusion_net(train_img)
				#print(train_SwinT_y)
				pre_train_ViT = torch.argmax(train_ViT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_ViT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_ViT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.to(gpu_device)
			val_label = val_label.to(gpu_device)
			with torch.no_grad():
				val_ViT_y, val_ViG_y, val_Fusion_y = fusion_net(val_img)
				val_ViT_loss = Loss_ViT(val_ViT_y, val_label)
				val_ViG_loss = Loss_ViG(val_ViG_y, val_label)
				val_Fusion_loss = Loss_Fusion(val_Fusion_y, val_label)
				pre_val_ViT = torch.argmax(val_ViT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_ViT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number),'\n',
			  'epoch ' + str(i + 1), ' train_ViT_loss:{:.4}'.format(loss_ViT.detach().cpu().numpy()),
			  ' train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
			  ' val_SViT_loss:{:.4}'.format(val_ViT_loss.detach().cpu().numpy()),
			  ' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)),
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
	test_ViT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.to(gpu_device)
		test_label = test_label.to(gpu_device)
		with torch.no_grad():
			test_ViT_y, test_ViG_y, test_Fusion_y = fusion_net(test_img)
			test_ViT_loss = Loss_ViT(test_ViT_y, test_label)
			pre_test_ViT = torch.argmax(test_ViT_y, dim=1)
			test_ViT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_ViT.detach().cpu().numpy()))
			test_VMamba_loss = Loss_ViG(test_ViG_y, test_label)
			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))
			test_Fusion_loss = Loss_Fusion(test_Fusion_y, test_label)
			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('########################## testing results #########################''\n',)
	print('train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
			' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)),
			' test_ViT_acc:{:.4}'.format(np.mean(test_ViT_acc)), '\n',
			'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
			' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
			'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
			' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = fusion_net.state_dict()
	torch.save(g, weight_path)
	return test_Fusion_acc


def train_moo(number=None, fusion_net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, fusion_weight=None,
			  vit_weight=None, vig_weight=None, weight_path=None):
	fusion_net = fusion_net.to(gpu_device)
	Loss_ViT = nn.CrossEntropyLoss()
	Loss_ViG = nn.CrossEntropyLoss()
	Loss_Fusion = nn.CrossEntropyLoss()

	for i in range(epoch):
		start_time = time.time()
		para = [{'params': fusion_net.stem.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.prediction.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.vig_out.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.block_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_4.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_5.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_6.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_7.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_8.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_9.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_10.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_11.parameters(), 'lr': vit_lr_schedule(i)},

				{'params': fusion_net.patch_embed.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.pos_drop.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_0.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_1.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_2.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_3.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_4.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_5.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_6.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_7.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_8.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_9.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_10.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_11.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.norm.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.head.parameters(), 'lr': vit_lr_schedule(i)}]
		optim = torch.optim.AdamW(para)
		fusion_net.train()
		for img_data, img_label in train_loader:
			img_data = img_data.to(gpu_device)
			img_label = img_label.to(gpu_device)

			vit_y, vig_y, fusion_y = fusion_net(img_data)

			loss_ViT = Loss_ViT(vit_y, img_label)
			loss_ViG = Loss_ViG(vig_y, img_label)
			loss_Fusion = Loss_Fusion(fusion_y, img_label)

			loss = fusion_weight * loss_Fusion + vit_weight * loss_ViT + vig_weight * loss_ViG

			loss.backward()
			optim.step()
			optim.zero_grad()

		fusion_net.eval()
		train_ViT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []

		for train_img, train_label in train_loader:
			train_img = train_img.to(gpu_device)
			train_label = train_label.to(gpu_device)
			with torch.no_grad():
				#print(train_label)
				train_ViT_y, train_ViG_y, train_Fusion_y = fusion_net(train_img)
				#print(train_SwinT_y)
				pre_train_ViT = torch.argmax(train_ViT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_ViT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_ViT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.to(gpu_device)
			val_label = val_label.to(gpu_device)
			with torch.no_grad():
				val_ViT_y, val_ViG_y, val_Fusion_y = fusion_net(val_img)
				val_ViT_loss = Loss_ViT(val_ViT_y, val_label)
				val_ViG_loss = Loss_ViG(val_ViG_y, val_label)
				val_Fusion_loss = Loss_Fusion(val_Fusion_y, val_label)
				val_loss_sum = fusion_weight * val_Fusion_loss + vit_weight * val_ViT_loss + vig_weight * val_ViG_loss
				pre_val_ViT = torch.argmax(val_ViT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_ViT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number),'\n',
			  'epoch ' + str(i + 1), ' train_ViT_loss:{:.4}'.format(loss_ViT.detach().cpu().numpy()),
			  ' train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
			  ' val_SViT_loss:{:.4}'.format(val_ViT_loss.detach().cpu().numpy()),
			  ' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)),
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

	fusion_net.eval()
	test_ViT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.to(gpu_device)
		test_label = test_label.to(gpu_device)
		with torch.no_grad():
			test_ViT_y, test_ViG_y, test_Fusion_y = fusion_net(test_img)
			test_ViT_loss = Loss_ViT(test_ViT_y, test_label)
			pre_test_ViT = torch.argmax(test_ViT_y, dim=1)
			test_ViT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_ViT.detach().cpu().numpy()))
			test_VMamba_loss = Loss_ViG(test_ViG_y, test_label)
			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))
			test_Fusion_loss = Loss_Fusion(test_Fusion_y, test_label)
			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('########################## testing results #########################''\n',)
	print('train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
			' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)),
			' test_ViT_acc:{:.4}'.format(np.mean(test_ViT_acc)), '\n',
			'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
			' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
			'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
			' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = fusion_net.state_dict()
	torch.save(g, weight_path)
	return test_Fusion_acc

def train_moo_fdaf(number=None, fusion_net=None, train_loader=None, val_loader=None, test_loader=None, epoch=None, gpu_device=None, fusion_weight=None,
			  vit_weight=None, vig_weight=None, weight_path=None):
	fusion_net = fusion_net.to(gpu_device)
	Loss_ViT = nn.CrossEntropyLoss()
	Loss_ViG = nn.CrossEntropyLoss()
	Loss_Fusion = nn.CrossEntropyLoss()

	for i in range(epoch):
		start_time = time.time()
		para = [{'params': fusion_net.stem.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.prediction.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.vig_out.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_0.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_3.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_4.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_5.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_6.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_7.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_8.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_9.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_10.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.block_11.parameters(), 'lr': vit_lr_schedule(i)},

				{'params': fusion_net.patch_embed.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.pos_drop.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_0.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_1.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_2.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_3.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_4.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_5.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_6.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_7.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_8.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_9.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_10.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.transformer_block_11.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.norm.parameters(), 'lr': cnn_lr_schedule(i)},
				{'params': fusion_net.head.parameters(), 'lr': cnn_lr_schedule(i)},

				{'params': fusion_net.fusion_block_dense_1.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.fusion_block_dense_2.parameters(), 'lr': vit_lr_schedule(i)},
				{'params': fusion_net.fusion_layer.parameters(), 'lr': vit_lr_schedule(i)},
				]
		optim = torch.optim.AdamW(para)
		fusion_net.train()
		for img_data, img_label in train_loader:
			img_data = img_data.to(gpu_device)
			img_label = img_label.to(gpu_device)

			vit_y, vig_y, fusion_y = fusion_net(img_data)

			loss_ViT = Loss_ViT(vit_y, img_label)
			loss_ViG = Loss_ViG(vig_y, img_label)
			loss_Fusion = Loss_Fusion(fusion_y, img_label)

			loss = fusion_weight * loss_Fusion + vit_weight * loss_ViT + vig_weight * loss_ViG

			loss.backward()
			optim.step()
			optim.zero_grad()

		fusion_net.eval()
		train_ViT_acc = []
		train_ViG_acc = []
		train_Fusion_acc = []

		for train_img, train_label in train_loader:
			train_img = train_img.to(gpu_device)
			train_label = train_label.to(gpu_device)
			with torch.no_grad():
				#print(train_label)
				train_ViT_y, train_ViG_y, train_Fusion_y = fusion_net(train_img)
				#print(train_SwinT_y)
				pre_train_ViT = torch.argmax(train_ViT_y, dim=1)
				pre_train_ViG = torch.argmax(train_ViG_y, dim=1)
				pre_train_Fusion = torch.argmax(train_Fusion_y, dim=1)
				train_ViT_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViT.detach().cpu().numpy()))
				train_ViG_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_ViG.detach().cpu().numpy()))
				train_Fusion_acc.append(accuracy_score(train_label.detach().cpu().numpy(), pre_train_Fusion.detach().cpu().numpy()))

		val_ViT_acc = []
		val_ViG_acc = []
		val_Fusion_acc = []
		# val_Sum_acc = []
		for val_img, val_label in val_loader:
			val_img = val_img.to(gpu_device)
			val_label = val_label.to(gpu_device)
			with torch.no_grad():
				val_ViT_y, val_ViG_y, val_Fusion_y = fusion_net(val_img)
				val_ViT_loss = Loss_ViT(val_ViT_y, val_label)
				val_ViG_loss = Loss_ViG(val_ViG_y, val_label)
				val_Fusion_loss = Loss_Fusion(val_Fusion_y, val_label)
				val_loss_sum = fusion_weight * val_Fusion_loss + vit_weight * val_ViT_loss + vig_weight * val_ViG_loss
				pre_val_ViT = torch.argmax(val_ViT_y, dim=1)
				pre_val_ViG = torch.argmax(val_ViG_y, dim=1)
				pre_val_Fusion = torch.argmax(val_Fusion_y, dim=1)
				val_ViT_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViT.detach().cpu().numpy()))
				val_ViG_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_ViG.detach().cpu().numpy()))
				val_Fusion_acc.append(accuracy_score(val_label.detach().cpu().numpy(), pre_val_Fusion.detach().cpu().numpy()))

		end_time = time.time()
		print('number ' + str(number),'\n',
			  'epoch ' + str(i + 1), ' train_ViT_loss:{:.4}'.format(loss_ViT.detach().cpu().numpy()),
			  ' train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
			  ' val_SViT_loss:{:.4}'.format(val_ViT_loss.detach().cpu().numpy()),
			  ' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)),
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

	fusion_net.eval()
	test_ViT_acc = []
	test_ViG_acc = []
	test_Fusion_acc = []
	for test_img, test_label in test_loader:
		test_img = test_img.to(gpu_device)
		test_label = test_label.to(gpu_device)
		with torch.no_grad():
			test_ViT_y, test_ViG_y, test_Fusion_y = fusion_net(test_img)
			test_ViT_loss = Loss_ViT(test_ViT_y, test_label)
			pre_test_ViT = torch.argmax(test_ViT_y, dim=1)
			test_ViT_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_ViT.detach().cpu().numpy()))
			test_VMamba_loss = Loss_ViG(test_ViG_y, test_label)
			pre_test_VMamba = torch.argmax(test_ViG_y, dim=1)
			test_ViG_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_VMamba.detach().cpu().numpy()))
			test_Fusion_loss = Loss_Fusion(test_Fusion_y, test_label)
			pre_test_Fusion = torch.argmax(test_Fusion_y, dim=1)
			test_Fusion_acc.append(accuracy_score(test_label.detach().cpu().numpy(), pre_test_Fusion.detach().cpu().numpy()))
	print('########################## testing results #########################''\n',)
	print('train_ViT_acc:{:.4}'.format(np.mean(train_ViT_acc)),
			' val_ViT_acc:{:.4}'.format(np.mean(val_ViT_acc)),
			' test_ViT_acc:{:.4}'.format(np.mean(test_ViT_acc)), '\n',
			'train_ViG_acc:{:.4}'.format(np.mean(train_ViG_acc)),
			' val_ViG_acc:{:.4}'.format(np.mean(val_ViG_acc)),
			' test_ViG_acc:{:.4}'.format(np.mean(test_ViG_acc)), '\n',
			'train_Fusion_acc:{:.4}'.format(np.mean(train_Fusion_acc)),
			' val_Fusion_acc:{:.4}'.format(np.mean(val_Fusion_acc)),
			' test_Fusion_acc:{:.4}'.format(np.mean(test_Fusion_acc)))
	g = fusion_net.state_dict()
	torch.save(g, weight_path)
	return test_Fusion_acc


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