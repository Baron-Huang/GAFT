import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_


class ViT2GNN(nn.Module):
	def __init__(self, swinT_base=None, vig_base=None, class_num=3):
		super(ViT2GNN, self).__init__()
		###############SwinT#################
		self.layers_0 = swinT_base.layers[0]
		self.layers_1 = swinT_base.layers[1]
		self.layers_2 = swinT_base.layers[2]
		self.layers_3 = swinT_base.layers[3]
		self.patch_embed = swinT_base.patch_embed
		self.pos_drop = swinT_base.pos_drop
		self.norm = swinT_base.norm
		self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
		self.head = swinT_base.head
		################ViG################
		self.stem = vig_base.stem
		self.prediction = vig_base.prediction[0:3]
		self.vig_out = nn.Conv2d(1024, 3, 1, 1)
		self.flat = nn.Flatten()
		self.block_0 = vig_base.backbone[0]
		self.block_1 = vig_base.backbone[1]
		self.block_2 = vig_base.backbone[2]
		self.block_3 = vig_base.backbone[3]
		self.block_4 = vig_base.backbone[4]
		self.block_5 = vig_base.backbone[5]
		self.block_6 = vig_base.backbone[6]
		self.block_7 = vig_base.backbone[7]
		self.block_8 = vig_base.backbone[8]
		self.block_9 = vig_base.backbone[9]
		self.block_10 = vig_base.backbone[10]
		self.block_11 = vig_base.backbone[11]


	def forward(self, x):
		#############################ViG stage0##########################
		vig_y = self.stem(x)
		vig_y = self.block_0(vig_y)
		vig_y_0 = self.block_1(vig_y)  # 16，192，14，14  stage0 out
		############################SwinT stage0##########################
		swin_t_y = self.patch_embed(x)
		swin_t_y = self.pos_drop(swin_t_y)
		swin_t_y_0 = self.layers_0(swin_t_y)  # 16，784，192
		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################SwinT stage1##########################
		swin_t_y_1 = self.layers_1(swin_t_y_0)  # 16，196，384
		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################SwinT stage2##########################
		swin_t_y_2 = self.layers_2(swin_t_y_1)  # 16，49，768
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + (u.permute(0, 2, 1) @ y)
		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################SwinT stage3##########################
		swin_t_y_3 = self.layers_3(swin_t_y_2)
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + self.w * (u.permute(0, 2, 1) @ y)
		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		swin_t_y = self.norm(swin_t_y_3)
		swin_t_y_fusion = self.avgp(swin_t_y.permute(0, 2, 1))
		# print('s',swin_t_y_fusion.permute(0, 2, 1).squeeze(1).shape)
		swin_t_y = self.head(swin_t_y_fusion.reshape(swin_t_y.shape[0], swin_t_y_fusion.shape[1]))

		fusion_out = swin_t_y + vig_y

		return fusion_out#swin_t_y, vig_y, fusion_out


class ViT2GNN_w(nn.Module):
	def __init__(self, swinT_base=None, vig_base=None, class_num=3):
		super(ViT2GNN_w, self).__init__()
		###############SwinT#################
		self.layers_0 = swinT_base.layers[0]
		self.layers_1 = swinT_base.layers[1]
		self.layers_2 = swinT_base.layers[2]
		self.layers_3 = swinT_base.layers[3]
		self.patch_embed = swinT_base.patch_embed
		self.pos_drop = swinT_base.pos_drop
		self.norm = swinT_base.norm
		self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
		self.head = swinT_base.head
		################ViG################
		self.stem = vig_base.stem
		self.prediction = vig_base.prediction[0:3]
		self.vig_out = nn.Conv2d(1024, 3, 1, 1)
		self.flat = nn.Flatten()
		self.block_0 = vig_base.backbone[0]
		self.block_1 = vig_base.backbone[1]
		self.block_2 = vig_base.backbone[2]
		self.block_3 = vig_base.backbone[3]
		self.block_4 = vig_base.backbone[4]
		self.block_5 = vig_base.backbone[5]
		self.block_6 = vig_base.backbone[6]
		self.block_7 = vig_base.backbone[7]
		self.block_8 = vig_base.backbone[8]
		self.block_9 = vig_base.backbone[9]
		self.block_10 = vig_base.backbone[10]
		self.block_11 = vig_base.backbone[11]


	def forward(self, x):
		#############################ViG stage0##########################
		vig_y = self.stem(x)
		vig_y = self.block_0(vig_y)
		vig_y_0 = self.block_1(vig_y)  # 16，192，14，14  stage0 out
		############################SwinT stage0##########################
		swin_t_y = self.patch_embed(x)
		swin_t_y = self.pos_drop(swin_t_y)
		swin_t_y_0 = self.layers_0(swin_t_y)  # 16，784，192
		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################SwinT stage1##########################
		swin_t_y_1 = self.layers_1(swin_t_y_0)  # 16，196，384
		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################SwinT stage2##########################
		swin_t_y_2 = self.layers_2(swin_t_y_1)  # 16，49，768
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + (u.permute(0, 2, 1) @ y)
		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################SwinT stage3##########################
		swin_t_y_3 = self.layers_3(swin_t_y_2)
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + self.w * (u.permute(0, 2, 1) @ y)
		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		swin_t_y = self.norm(swin_t_y_3)
		swin_t_y_fusion = self.avgp(swin_t_y.permute(0, 2, 1))
		# print('s',swin_t_y_fusion.permute(0, 2, 1).squeeze(1).shape)
		swin_t_y = self.head(swin_t_y_fusion.reshape(swin_t_y.shape[0], swin_t_y_fusion.shape[1]))

		fusion_out = swin_t_y + vig_y

		return swin_t_y, vig_y, fusion_out, swin_t_y_fusion.squeeze(), vig_y_fusion


def init_weights(module):
	if isinstance(module, nn.Linear):
		kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
		if module.bias is not None:
			constant_(module.bias, 0)
	elif isinstance(module, nn.Conv2d):
		kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
		if module.bias is not None:
			constant_(module.bias, 0)
	print('init init')

class ViT2GNN_AB(nn.Module):
	def __init__(self, swinT_base=None, vig_base=None, class_num=3):
		super(ViT2GNN_AB, self).__init__()
		###############SwinT#################
		self.layers_0 = swinT_base.layers[0]
		self.layers_1 = swinT_base.layers[1]
		self.layers_2 = swinT_base.layers[2]
		self.layers_3 = swinT_base.layers[3]
		self.patch_embed = swinT_base.patch_embed
		self.pos_drop = swinT_base.pos_drop
		self.norm = swinT_base.norm
		self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
		self.head = swinT_base.head
		################ViG################
		self.stem = vig_base.stem
		self.prediction = vig_base.prediction[0:3]
		self.vig_out = nn.Conv2d(1024, 3, 1, 1)
		self.flat = nn.Flatten()
		self.block_0 = vig_base.backbone[0]
		self.block_1 = vig_base.backbone[1]
		self.block_2 = vig_base.backbone[2]
		self.block_3 = vig_base.backbone[3]
		self.block_4 = vig_base.backbone[4]
		self.block_5 = vig_base.backbone[5]
		self.block_6 = vig_base.backbone[6]
		self.block_7 = vig_base.backbone[7]
		self.block_8 = vig_base.backbone[8]
		self.block_9 = vig_base.backbone[9]
		self.block_10 = vig_base.backbone[10]
		self.block_11 = vig_base.backbone[11]

		self.Dims_VtoS_0 = nn.Sequential(
			ChannelAttention(192, feature_size=14),
			nn.ConvTranspose2d(192, 192, 2, 2, 0),
			nn.Flatten(start_dim=2, end_dim=3),
			TransposeLayer(),
			nn.LayerNorm([784, 192])
		)
		self.Dims_StoV_0 = nn.Sequential(
			TransposeLayer(),
			ViewBlock(),
			nn.Conv2d(192, 192, kernel_size=2, stride=2, padding=0),
			PositionAttention(192),
			nn.LayerNorm([192, 14, 14])
		)

		self.Dims_VtoS_1 = nn.Sequential(
			ChannelAttention(192, feature_size=14),
			nn.Conv2d(192, 384, 1),
			nn.Flatten(start_dim=2, end_dim=3),
			TransposeLayer(),
			nn.LayerNorm([196, 384])
		)
		self.Dims_StoV_1 = nn.Sequential(
			TransposeLayer(),
			ViewBlock(),
			nn.ConvTranspose2d(384, 192, kernel_size=1),
			PositionAttention(192),
			nn.LayerNorm([192, 14, 14])
		)

		self.Dims_VtoS_2 = nn.Sequential(
			nn.AvgPool2d(kernel_size=2, stride=2),
			ChannelAttention(192,feature_size=7),
			nn.Conv2d(192, 768, 1),
			nn.Flatten(start_dim=2, end_dim=3),
			TransposeLayer(),
			nn.LayerNorm([49, 768])
		)
		self.Dims_StoV_2 = nn.Sequential(
			TransposeLayer(),
			ViewBlock(),
			nn.ConvTranspose2d(768, 192, kernel_size=2, stride=2, padding=0),
			PositionAttention(192),
			nn.LayerNorm([192, 14, 14])
		)

		# init_weights(self.Dims_VtoS_0)
		# init_weights(self.Dims_VtoS_1)
		# init_weights(self.Dims_VtoS_2)
		# init_weights(self.Dims_StoV_0)
		# init_weights(self.Dims_StoV_1)
		# init_weights(self.Dims_StoV_2)

	def forward(self, x):
		#############################ViG stage0##########################
		vig_y = self.stem(x)
		vig_y = self.block_0(vig_y)
		vig_y_0 = self.block_1(vig_y)  # 16，192，14，14  stage0 out
		############################SwinT stage0##########################
		swin_t_y = self.patch_embed(x)
		swin_t_y = self.pos_drop(swin_t_y)
		swin_t_y_0 = self.layers_0(swin_t_y)  # 16，784，192
		#############################Attention##########################
		vig_y_VtoS_0 = self.Dims_VtoS_0(vig_y_0)  # (1,784,192)
		swin_t_y_0 = swin_t_y_0 + vig_y_VtoS_0
		swin_t_y_StoV_0 = self.Dims_StoV_0(swin_t_y_0)
		vig_y_0 = vig_y_0 + swin_t_y_StoV_0

		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################SwinT stage1##########################
		swin_t_y_1 = self.layers_1(swin_t_y_0)  # 16，196，384
		#############################Attention##########################
		vig_y_VtoS_1 = self.Dims_VtoS_1(vig_y_1)  # (1,784,192)
		swin_t_y_1 = swin_t_y_1 + vig_y_VtoS_1
		swin_t_y_StoV_1 = self.Dims_StoV_1(swin_t_y_1)
		vig_y_2 = vig_y_1 + swin_t_y_StoV_1

		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################SwinT stage2##########################
		swin_t_y_2 = self.layers_2(swin_t_y_1)  # 16，49，768
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + (u.permute(0, 2, 1) @ y)
		#############################Attention##########################
		vig_y_VtoS_2 = self.Dims_VtoS_2(vig_y_2)  # (1,784,192)
		swin_t_y_2 = swin_t_y_2 + vig_y_VtoS_2
		swin_t_y_StoV_2 = self.Dims_StoV_2(swin_t_y_2)
		vig_y_2 = vig_y_2 + swin_t_y_StoV_2

		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################SwinT stage3##########################
		swin_t_y_3 = self.layers_3(swin_t_y_2)
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + self.w * (u.permute(0, 2, 1) @ y)
		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		swin_t_y = self.norm(swin_t_y_3)
		swin_t_y_fusion = self.avgp(swin_t_y.permute(0, 2, 1))
		# print('s',swin_t_y_fusion.permute(0, 2, 1).squeeze(1).shape)
		swin_t_y = self.head(swin_t_y_fusion.reshape(swin_t_y.shape[0], swin_t_y_fusion.shape[1]))

		fusion_out = swin_t_y + vig_y

		return swin_t_y, vig_y, fusion_out

class ViT2GNN_FDAF(nn.Module):
	def __init__(self, swinT_base=None, vig_base=None, class_num=3):
		super(ViT2GNN_FDAF, self).__init__()
		###############SwinT#################
		self.layers_0 = swinT_base.layers[0]
		self.layers_1 = swinT_base.layers[1]
		self.layers_2 = swinT_base.layers[2]
		self.layers_3 = swinT_base.layers[3]
		self.patch_embed = swinT_base.patch_embed
		self.pos_drop = swinT_base.pos_drop
		self.norm = swinT_base.norm
		self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
		self.head = swinT_base.head
		################ViG################
		self.stem = vig_base.stem
		self.prediction = vig_base.prediction[0:3]
		self.vig_out = nn.Conv2d(1024, 3, 1, 1)
		self.flat = nn.Flatten()
		self.block_0 = vig_base.backbone[0]
		self.block_1 = vig_base.backbone[1]
		self.block_2 = vig_base.backbone[2]
		self.block_3 = vig_base.backbone[3]
		self.block_4 = vig_base.backbone[4]
		self.block_5 = vig_base.backbone[5]
		self.block_6 = vig_base.backbone[6]
		self.block_7 = vig_base.backbone[7]
		self.block_8 = vig_base.backbone[8]
		self.block_9 = vig_base.backbone[9]
		self.block_10 = vig_base.backbone[10]
		self.block_11 = vig_base.backbone[11]

		self.fusion_block_dense_1 = nn.Linear(1792, 384)
		self.fusion_block_dense_2 = nn.Linear(384, 1792)
		self.fusion_layer = nn.Linear(1792, class_num)

	def forward(self, x):
		#############################ViG stage0##########################
		vig_y = self.stem(x)
		vig_y = self.block_0(vig_y)
		vig_y_0 = self.block_1(vig_y)  # 16，192，14，14  stage0 out
		############################SwinT stage0##########################
		swin_t_y = self.patch_embed(x)
		swin_t_y = self.pos_drop(swin_t_y)
		swin_t_y_0 = self.layers_0(swin_t_y)  # 16，784，192
		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################SwinT stage1##########################
		swin_t_y_1 = self.layers_1(swin_t_y_0)  # 16，196，384
		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################SwinT stage2##########################
		swin_t_y_2 = self.layers_2(swin_t_y_1)  # 16，49，768
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + (u.permute(0, 2, 1) @ y)
		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################SwinT stage3##########################
		swin_t_y_3 = self.layers_3(swin_t_y_2)
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + self.w * (u.permute(0, 2, 1) @ y)
		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		swin_t_y = self.norm(swin_t_y_3)
		swin_t_y_fusion = self.avgp(swin_t_y.permute(0, 2, 1))
		# print('s',swin_t_y_fusion.permute(0, 2, 1).squeeze(1).shape)
		swin_t_y = self.head(swin_t_y_fusion.reshape(swin_t_y.shape[0], swin_t_y_fusion.shape[1]))

		fusion_out = torch.cat((vig_y_fusion, swin_t_y_fusion.squeeze(-1)), dim=1)
		fusion_weights = self.fusion_block_dense_1(fusion_out)
		fusion_weights = torch.nn.functional.relu(fusion_weights)
		fusion_weights = self.fusion_block_dense_2(fusion_weights)
		fusion_weights = torch.nn.functional.sigmoid(fusion_weights)
		fusion_out = fusion_out * fusion_weights
		fusion_out = self.fusion_layer(fusion_out)


		return swin_t_y, vig_y, fusion_out

class ViT2GNN_AAMFF(nn.Module):
	def __init__(self, swinT_base=None, vig_base=None, class_num=3):
		super(ViT2GNN_AAMFF, self).__init__()
		###############SwinT#################
		self.layers_0 = swinT_base.layers[0]
		self.layers_1 = swinT_base.layers[1]
		self.layers_2 = swinT_base.layers[2]
		self.layers_3 = swinT_base.layers[3]
		self.patch_embed = swinT_base.patch_embed
		self.pos_drop = swinT_base.pos_drop
		self.norm = swinT_base.norm
		self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
		self.head = swinT_base.head
		################ViG################
		self.stem = vig_base.stem
		self.prediction = vig_base.prediction[0:3]
		self.vig_out = nn.Conv2d(1024, 3, 1, 1)
		self.flat = nn.Flatten()
		self.block_0 = vig_base.backbone[0]
		self.block_1 = vig_base.backbone[1]
		self.block_2 = vig_base.backbone[2]
		self.block_3 = vig_base.backbone[3]
		self.block_4 = vig_base.backbone[4]
		self.block_5 = vig_base.backbone[5]
		self.block_6 = vig_base.backbone[6]
		self.block_7 = vig_base.backbone[7]
		self.block_8 = vig_base.backbone[8]
		self.block_9 = vig_base.backbone[9]
		self.block_10 = vig_base.backbone[10]
		self.block_11 = vig_base.backbone[11]

		self.swin_fc1 = nn.Linear(1792, 2048)
		self.swin_fc2 = nn.Linear(2048, 768)
		self.vig_fc1 = nn.Linear(1792, 2048)
		self.vig_fc2 = nn.Linear(2048, 1024)
		self.fusion_fc1 = nn.Linear(1792, 512)
		self.fusion_fc2 = nn.Linear(512, class_num)

		self.swin_id=nn.Identity()
		self.vig_id = nn.Identity()

	def forward(self, x):
		#############################ViG stage0##########################
		vig_y = self.stem(x)
		vig_y = self.block_0(vig_y)
		vig_y_0 = self.block_1(vig_y)  # 16，192，14，14  stage0 out
		############################SwinT stage0##########################
		swin_t_y = self.patch_embed(x)
		swin_t_y = self.pos_drop(swin_t_y)
		swin_t_y_0 = self.layers_0(swin_t_y)  # 16，784，192
		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################SwinT stage1##########################
		swin_t_y_1 = self.layers_1(swin_t_y_0)  # 16，196，384
		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################SwinT stage2##########################
		swin_t_y_2 = self.layers_2(swin_t_y_1)  # 16，49，768
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + (u.permute(0, 2, 1) @ y)
		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################SwinT stage3##########################
		swin_t_y_3 = self.layers_3(swin_t_y_2)
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + self.w * (u.permute(0, 2, 1) @ y)
		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		swin_t_y = self.norm(swin_t_y_3)
		swin_t_y_fusion = self.avgp(swin_t_y.permute(0, 2, 1))
		# print('s',swin_t_y_fusion.permute(0, 2, 1).squeeze(1).shape)
		swin_t_y = self.head(swin_t_y_fusion.reshape(swin_t_y.shape[0], swin_t_y_fusion.shape[1]))

		fusion_out = torch.cat((vig_y_fusion, swin_t_y_fusion.squeeze(-1)), dim=1)

		swin_fusion_out = self.swin_fc1(fusion_out)
		swin_fusion_out = torch.nn.functional.relu(swin_fusion_out)
		swin_fusion_out = self.swin_fc2(swin_fusion_out)
		swin_fusion_out_weight = torch.nn.functional.sigmoid(swin_fusion_out)
		swin_fusion_out = swin_t_y_fusion.squeeze(-1) * swin_fusion_out_weight
		swin_fusion_out = self.swin_id(swin_fusion_out)

		vig_fusion_out = self.vig_fc1(fusion_out)
		vig_fusion_out = torch.nn.functional.relu(vig_fusion_out)
		vig_fusion_out = self.vig_fc2(vig_fusion_out)
		vig_fusion_out_weight = torch.nn.functional.sigmoid(vig_fusion_out)
		vig_fusion_out = vig_fusion_out_weight * vig_y_fusion
		vig_fusion_out = self.vig_id(vig_fusion_out)

		fusion_out = torch.cat((swin_fusion_out, vig_fusion_out), dim=1)
		fusion_out = self.fusion_fc1(fusion_out)
		fusion_out = torch.nn.functional.relu(fusion_out)
		fusion_out = self.fusion_fc2(fusion_out)
		return swin_t_y, vig_y, fusion_out, swin_t_y, swin_t_y


class ViT2GNN_AAMFF_Dis(nn.Module):
	def __init__(self, swinT_base=None, vig_base=None, class_num=3):
		super(ViT2GNN_AAMFF_Dis, self).__init__()
		###############SwinT#################
		self.layers_0 = swinT_base.layers[0]
		self.layers_1 = swinT_base.layers[1]
		self.layers_2 = swinT_base.layers[2]
		self.layers_3 = swinT_base.layers[3]
		self.patch_embed = swinT_base.patch_embed
		self.pos_drop = swinT_base.pos_drop
		self.norm = swinT_base.norm
		self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
		self.head = swinT_base.head
		################ViG################
		self.stem = vig_base.stem
		self.prediction = vig_base.prediction[0:3]
		self.vig_out = nn.Conv2d(1024, 3, 1, 1)
		self.flat = nn.Flatten()
		self.block_0 = vig_base.backbone[0]
		self.block_1 = vig_base.backbone[1]
		self.block_2 = vig_base.backbone[2]
		self.block_3 = vig_base.backbone[3]
		self.block_4 = vig_base.backbone[4]
		self.block_5 = vig_base.backbone[5]
		self.block_6 = vig_base.backbone[6]
		self.block_7 = vig_base.backbone[7]
		self.block_8 = vig_base.backbone[8]
		self.block_9 = vig_base.backbone[9]
		self.block_10 = vig_base.backbone[10]
		self.block_11 = vig_base.backbone[11]

		self.swin_fc1 = nn.Linear(1792, 2048)
		self.swin_fc2 = nn.Linear(2048, 768)
		self.vig_fc1 = nn.Linear(1792, 2048)
		self.vig_fc2 = nn.Linear(2048, 1024)
		self.fusion_fc1 = nn.Linear(1792, 512)
		self.fusion_fc2 = nn.Linear(512, class_num)

		self.swin_id = nn.Identity()
		self.vig_id = nn.Identity()
	def forward(self, x):
		#############################ViG stage0##########################
		vig_y = self.stem(x)
		vig_y = self.block_0(vig_y)
		vig_y_0 = self.block_1(vig_y)  # 16，192，14，14  stage0 out
		############################SwinT stage0##########################
		swin_t_y = self.patch_embed(x)
		swin_t_y = self.pos_drop(swin_t_y)
		swin_t_y_0 = self.layers_0(swin_t_y)  # 16，784，192
		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################SwinT stage1##########################
		swin_t_y_1 = self.layers_1(swin_t_y_0)  # 16，196，384
		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################SwinT stage2##########################
		swin_t_y_2 = self.layers_2(swin_t_y_1)  # 16，49，768
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + (u.permute(0, 2, 1) @ y)
		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################SwinT stage3##########################
		swin_t_y_3 = self.layers_3(swin_t_y_2)
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + self.w * (u.permute(0, 2, 1) @ y)
		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		swin_t_y = self.norm(swin_t_y_3)
		swin_t_y_fusion = self.avgp(swin_t_y.permute(0, 2, 1))
		# print('s',swin_t_y_fusion.permute(0, 2, 1).squeeze(1).shape)
		swin_t_y = self.head(swin_t_y_fusion.reshape(swin_t_y.shape[0], swin_t_y_fusion.shape[1]))

		fusion_out = torch.cat((vig_y_fusion, swin_t_y_fusion.squeeze(-1)), dim=1)

		swin_fusion_out = self.swin_fc1(fusion_out)
		swin_fusion_out = torch.nn.functional.relu(swin_fusion_out)
		swin_fusion_out = self.swin_fc2(swin_fusion_out)
		swin_fusion_out_weight = torch.nn.functional.sigmoid(swin_fusion_out)
		swin_fusion_out = swin_t_y_fusion.squeeze(-1) * swin_fusion_out_weight
		swin_fusion_out = self.swin_id(swin_fusion_out)

		vig_fusion_out = self.vig_fc1(fusion_out)
		vig_fusion_out = torch.nn.functional.relu(vig_fusion_out)
		vig_fusion_out = self.vig_fc2(vig_fusion_out)
		vig_fusion_out_weight = torch.nn.functional.sigmoid(vig_fusion_out)
		vig_fusion_out = vig_fusion_out_weight * vig_y_fusion
		vig_fusion_out = self.vig_id(vig_fusion_out)

		fusion_out = torch.cat((swin_fusion_out, vig_fusion_out), dim=1)

		fusion_out = self.fusion_fc1(fusion_out)
		fusion_out = torch.nn.functional.relu(fusion_out)
		fusion_out = self.fusion_fc2(fusion_out)
		return swin_t_y, vig_y, fusion_out,swin_t_y_fusion, vig_y_fusion

class ViT2GNN_FDAF_Dis(nn.Module):
	def __init__(self, swinT_base=None, vig_base=None, class_num=3):
		super(ViT2GNN_FDAF_Dis, self).__init__()
		###############SwinT#################
		self.layers_0 = swinT_base.layers[0]
		self.layers_1 = swinT_base.layers[1]
		self.layers_2 = swinT_base.layers[2]
		self.layers_3 = swinT_base.layers[3]
		self.patch_embed = swinT_base.patch_embed
		self.pos_drop = swinT_base.pos_drop
		self.norm = swinT_base.norm
		self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
		self.head = swinT_base.head
		################ViG################
		self.stem = vig_base.stem
		self.prediction = vig_base.prediction[0:3]
		self.vig_out = nn.Conv2d(1024, 3, 1, 1)
		self.flat = nn.Flatten()
		self.block_0 = vig_base.backbone[0]
		self.block_1 = vig_base.backbone[1]
		self.block_2 = vig_base.backbone[2]
		self.block_3 = vig_base.backbone[3]
		self.block_4 = vig_base.backbone[4]
		self.block_5 = vig_base.backbone[5]
		self.block_6 = vig_base.backbone[6]
		self.block_7 = vig_base.backbone[7]
		self.block_8 = vig_base.backbone[8]
		self.block_9 = vig_base.backbone[9]
		self.block_10 = vig_base.backbone[10]
		self.block_11 = vig_base.backbone[11]

		self.fusion_block_dense_1 = nn.Linear(1792, 112)
		self.fusion_block_dense_2 = nn.Linear(112, 1792)
		self.fusion_layer = nn.Linear(1792, class_num)


	def forward(self, x):
		#############################ViG stage0##########################
		vig_y = self.stem(x)
		vig_y = self.block_0(vig_y)
		vig_y_0 = self.block_1(vig_y)  # 16，192，14，14  stage0 out
		############################SwinT stage0##########################
		swin_t_y = self.patch_embed(x)
		swin_t_y = self.pos_drop(swin_t_y)
		swin_t_y_0 = self.layers_0(swin_t_y)  # 16，784，192
		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################SwinT stage1##########################
		swin_t_y_1 = self.layers_1(swin_t_y_0)  # 16，196，384
		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################SwinT stage2##########################
		swin_t_y_2 = self.layers_2(swin_t_y_1)  # 16，49，768
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + (u.permute(0, 2, 1) @ y

		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################SwinT stage3##########################
		swin_t_y_3 = self.layers_3(swin_t_y_2)
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + self.w * (u.permute(0, 2, 1) @ y)
		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		swin_t_y = self.norm(swin_t_y_3)
		swin_t_y_fusion = self.avgp(swin_t_y.permute(0, 2, 1))
		# print('s',swin_t_y_fusion.permute(0, 2, 1).squeeze(1).shape)
		swin_t_y = self.head(swin_t_y_fusion.reshape(swin_t_y.shape[0], swin_t_y_fusion.shape[1]))

		fusion_out = torch.cat((vig_y_fusion, swin_t_y_fusion.squeeze(-1)), dim=1)
		fusion_weights = self.fusion_block_dense_1(fusion_out)
		fusion_weights = torch.nn.functional.relu(fusion_weights)
		fusion_weights = self.fusion_block_dense_2(fusion_weights)
		#fusion_weights = torch.nn.functional.softmax(fusion_weights, dim=1)
		fusion_out = fusion_out * fusion_weights
		fusion_out = self.fusion_layer(fusion_out)


		return swin_t_y, vig_y, fusion_out, swin_t_y_fusion, vig_y_fusion

class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=16, feature_size=None):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.MaxPool2d(feature_size)#nn.AdaptiveMaxPool2d(1)

		self.shared_MLP = nn.Sequential(
			nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
			nn.ReLU(),
			nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
		)
		# self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
		# self.relu1 = nn.ReLU()
		# self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out =self.shared_MLP(self.avg_pool(x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out =self.shared_MLP(self.max_pool(x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		out = avg_out + max_out
		x = x * self.sigmoid(out) #self.sigmoid(out) weights
		return x


class PositionAttention(nn.Module):
	""" Position attention module"""

	def __init__(self, in_channels, **kwargs):
		super(PositionAttention, self).__init__()
		self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
		self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
		self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
		self.alpha = nn.Parameter(torch.zeros(1))
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		batch_size, _, height, width = x.size()
		feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
		feat_c = self.conv_c(x).view(batch_size, -1, height * width)
		attention_s = self.softmax(torch.bmm(feat_b, feat_c))
		feat_d = self.conv_d(x).view(batch_size, -1, height * width)
		feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
		out = self.alpha * feat_e + x

		return out

class TransposeLayer(nn.Module):
	def forward(self, x):
		return torch.transpose(x, 1, 2)

class ViewBlock(nn.Module):
	def __init__(self):
		super(ViewBlock, self).__init__()

	def forward(self, x):
		batch_size, num_channels, a = x.size()
		a_sqrt = int(a ** 0.5)  # 计算a的平方根
		return x.view(batch_size, num_channels, a_sqrt, a_sqrt)


class ViT2GNN_AAMFF_cont(nn.Module):
	def __init__(self, swinT_base=None, vig_base=None, class_num=3):
		super(ViT2GNN_AAMFF_cont, self).__init__()
		###############SwinT#################
		self.layers_0 = swinT_base.layers[0]
		self.layers_1 = swinT_base.layers[1]
		self.layers_2 = swinT_base.layers[2]
		self.layers_3 = swinT_base.layers[3]
		self.patch_embed = swinT_base.patch_embed
		self.pos_drop = swinT_base.pos_drop
		self.norm = swinT_base.norm
		self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
		self.head = swinT_base.head
		################ViG################
		self.stem = vig_base.stem
		self.prediction = vig_base.prediction[0:3]
		self.vig_out = nn.Conv2d(1024, 3, 1, 1)
		self.flat = nn.Flatten()
		self.block_0 = vig_base.backbone[0]
		self.block_1 = vig_base.backbone[1]
		self.block_2 = vig_base.backbone[2]
		self.block_3 = vig_base.backbone[3]
		self.block_4 = vig_base.backbone[4]
		self.block_5 = vig_base.backbone[5]
		self.block_6 = vig_base.backbone[6]
		self.block_7 = vig_base.backbone[7]
		self.block_8 = vig_base.backbone[8]
		self.block_9 = vig_base.backbone[9]
		self.block_10 = vig_base.backbone[10]
		self.block_11 = vig_base.backbone[11]

		self.fusion_fc = nn.Linear(1792, class_num)

	def forward(self, x):
		#############################ViG stage0##########################
		vig_y = self.stem(x)
		vig_y = self.block_0(vig_y)
		vig_y_0 = self.block_1(vig_y)  # 16，192，14，14  stage0 out
		############################SwinT stage0##########################
		swin_t_y = self.patch_embed(x)
		swin_t_y = self.pos_drop(swin_t_y)
		swin_t_y_0 = self.layers_0(swin_t_y)  # 16，784，192
		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################SwinT stage1##########################
		swin_t_y_1 = self.layers_1(swin_t_y_0)  # 16，196，384
		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################SwinT stage2##########################
		swin_t_y_2 = self.layers_2(swin_t_y_1)  # 16，49，768
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + (u.permute(0, 2, 1) @ y)
		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################SwinT stage3##########################
		swin_t_y_3 = self.layers_3(swin_t_y_2)
		# y_sym = y @ y.permute(0, 2, 1)
		# _, u = torch.linalg.eigh(y_sym)
		# y = y + self.w * (u.permute(0, 2, 1) @ y)
		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		swin_t_y = self.norm(swin_t_y_3)
		swin_t_y_fusion = self.avgp(swin_t_y.permute(0, 2, 1))
		# print('s',swin_t_y_fusion.permute(0, 2, 1).squeeze(1).shape)
		swin_t_y = self.head(swin_t_y_fusion.reshape(swin_t_y.shape[0], swin_t_y_fusion.shape[1]))

		fusion_out = torch.cat((vig_y_fusion, swin_t_y_fusion.squeeze(-1)), dim=1)

		fusion_out = self.fusion_fc(fusion_out)
		return swin_t_y, vig_y, fusion_out

class Concatenation_Fusion(nn.Module):
    """Feature-level concatenation baseline for SwinT and ViG.

    The two pretrained backbones generate global 768-D and 1024-D features.
    Their features are concatenated along dim=1 and classified by a small MLP.
    """
    def __init__(self, swinT_base=None, vig_base=None, class_num=3, hidden_dim=512, dropout=0.2):
        super().__init__()
        if swinT_base is None or vig_base is None:
            raise ValueError('swinT_base and vig_base must both be provided.')

        self.swin_backbone = swinT_base
        self.vig_stem = vig_base.stem
        self.vig_pos_embed = vig_base.pos_embed
        self.vig_blocks = vig_base.backbone
        self.vig_num_blocks = vig_base.n_blocks
        # Keep the pretrained ViG feature projection and classification head.
        self.vig_projection = vig_base.prediction[0:3]
        self.vig_head = vig_base.prediction[4]

        swin_dim = getattr(swinT_base, 'num_features', 768)
        vig_dim = 1024
        self.fusion_classifier = nn.Sequential(
            nn.Linear(swin_dim + vig_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, class_num),
        )
        nn.init.trunc_normal_(self.fusion_classifier[0].weight, std=.02)
        nn.init.zeros_(self.fusion_classifier[0].bias)
        nn.init.trunc_normal_(self.fusion_classifier[3].weight, std=.02)
        nn.init.zeros_(self.fusion_classifier[3].bias)

    def _vig_features(self, x):
        x = self.vig_stem(x) + self.vig_pos_embed
        for index in range(self.vig_num_blocks):
            x = self.vig_blocks[index](x)
        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(self.vig_projection(x), 1)

    def forward(self, x):
        swin_features = self.swin_backbone.forward_features(x)
        swin_logits = self.swin_backbone.head(swin_features)

        vig_features = self._vig_features(x)
        vig_logits = self.vig_head(vig_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        fused_features = torch.cat((swin_features, vig_features), dim=1)
        fusion_logits = self.fusion_classifier(fused_features)
        return swin_logits, vig_logits, fusion_logits


class Cross_Attetion_Fusion(nn.Module):
    """Token-level cross-attention fusion baseline for SwinT and ViG.

    SwinT final tokens are queries; projected ViG final tokens are keys and values.
    """
    def __init__(self, swinT_base=None, vig_base=None, class_num=3, num_heads=8, dropout=0.2):
        super().__init__()
        if swinT_base is None or vig_base is None:
            raise ValueError('swinT_base and vig_base must both be provided.')

        self.swin_backbone = swinT_base
        self.vig_stem = vig_base.stem
        self.vig_pos_embed = vig_base.pos_embed
        self.vig_blocks = vig_base.backbone
        self.vig_num_blocks = vig_base.n_blocks
        self.vig_projection = vig_base.prediction[0:3]
        self.vig_head = vig_base.prediction[4]

        swin_dim = getattr(swinT_base, 'num_features', 768)
        self.vig_to_swin = nn.Linear(1024, swin_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=swin_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(swin_dim)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(swin_dim, swin_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(swin_dim // 2, class_num),
        )
        for layer in (self.vig_to_swin, self.fusion_classifier[0], self.fusion_classifier[3]):
            nn.init.trunc_normal_(layer.weight, std=.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _swin_tokens(self, x):
        x = self.swin_backbone.patch_embed(x)
        if self.swin_backbone.ape:
            x = x + self.swin_backbone.absolute_pos_embed
        x = self.swin_backbone.pos_drop(x)
        for layer in self.swin_backbone.layers:
            x = layer(x)
        return self.swin_backbone.norm(x)

    def _vig_feature_map(self, x):
        x = self.vig_stem(x) + self.vig_pos_embed
        for index in range(self.vig_num_blocks):
            x = self.vig_blocks[index](x)
        return x

    def _vig_features(self, feature_map):
        feature_map = F.adaptive_avg_pool2d(feature_map, 1)
        return torch.flatten(self.vig_projection(feature_map), 1)

    def _vig_tokens(self, feature_map):
        # 14x14 ViG features are pooled to 7x7, matching SwinT's 49 final tokens.
        feature_map = F.adaptive_avg_pool2d(feature_map, (7, 7))
        feature_map = self.vig_projection(feature_map)
        return feature_map.flatten(2).transpose(1, 2)

    def forward(self, x):
        swin_tokens = self._swin_tokens(x)
        swin_features = swin_tokens.mean(dim=1)
        swin_logits = self.swin_backbone.head(swin_features)

        vig_feature_map = self._vig_feature_map(x)
        vig_features = self._vig_features(vig_feature_map)
        vig_logits = self.vig_head(vig_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        vig_tokens = self.vig_to_swin(self._vig_tokens(vig_feature_map))
        attended_tokens, _ = self.cross_attention(
            query=swin_tokens, key=vig_tokens, value=vig_tokens, need_weights=False
        )
        fused_features = self.cross_norm(swin_tokens + attended_tokens).mean(dim=1)
        fusion_logits = self.fusion_classifier(fused_features)
        return swin_logits, vig_logits, fusion_logits


# Correctly spelled alias for use in external code.
Cross_Attention_Fusion = Cross_Attetion_Fusion

class Late_Fusion(nn.Module):
    """Logit-level late fusion baseline for SwinT and ViG.

    Each pretrained backbone makes an independent class prediction.  A learned,
    softmax-normalized pair of global weights combines their logits, which keeps
    the fusion decision transparent while avoiding intermediate-feature fusion.
    """
    def __init__(self, swinT_base=None, vig_base=None, class_num=3):
        super().__init__()
        if swinT_base is None or vig_base is None:
            raise ValueError('swinT_base and vig_base must both be provided.')

        self.swin_backbone = swinT_base
        self.vig_stem = vig_base.stem
        self.vig_pos_embed = vig_base.pos_embed
        self.vig_blocks = vig_base.backbone
        self.vig_num_blocks = vig_base.n_blocks
        self.vig_projection = vig_base.prediction[0:3]
        self.vig_head = vig_base.prediction[4]
        self.class_num = class_num
        # Equal initial contribution; weights are learned from the training data.
        self.fusion_weights = nn.Parameter(torch.zeros(2))

    def _vig_logits(self, x):
        x = self.vig_stem(x) + self.vig_pos_embed
        for index in range(self.vig_num_blocks):
            x = self.vig_blocks[index](x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.vig_projection(x)
        return self.vig_head(x).squeeze(-1).squeeze(-1)

    def forward(self, x):
        swin_features = self.swin_backbone.forward_features(x)
        swin_logits = self.swin_backbone.head(swin_features)
        vig_logits = self._vig_logits(x)
        weights = torch.softmax(self.fusion_weights, dim=0)
        fusion_logits = weights[0] * swin_logits + weights[1] * vig_logits
        return swin_logits, vig_logits, fusion_logits

class Weighted_Summation(nn.Module):
    """Feature-level weighted-summation baseline for SwinT and ViG.

    The two global embeddings are projected to a shared space, normalized, and
    combined using learned softmax weights before a final classifier.
    """
    def __init__(self, swinT_base=None, vig_base=None, class_num=3, fusion_dim=512, dropout=0.2):
        super().__init__()
        if swinT_base is None or vig_base is None:
            raise ValueError('swinT_base and vig_base must both be provided.')

        self.swin_backbone = swinT_base
        self.vig_stem = vig_base.stem
        self.vig_pos_embed = vig_base.pos_embed
        self.vig_blocks = vig_base.backbone
        self.vig_num_blocks = vig_base.n_blocks
        self.vig_projection = vig_base.prediction[0:3]
        self.vig_head = vig_base.prediction[4]

        swin_dim = getattr(swinT_base, 'num_features', 768)
        self.swin_projector = nn.Sequential(nn.Linear(swin_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.vig_projector = nn.Sequential(nn.Linear(1024, fusion_dim), nn.LayerNorm(fusion_dim))
        self.fusion_weights = nn.Parameter(torch.zeros(2))
        self.fusion_classifier = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, class_num),
        )
        for layer in (self.swin_projector[0], self.vig_projector[0], self.fusion_classifier[2]):
            nn.init.trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)

    def _vig_features(self, x):
        x = self.vig_stem(x) + self.vig_pos_embed
        for index in range(self.vig_num_blocks):
            x = self.vig_blocks[index](x)
        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(self.vig_projection(x), 1)

    def forward(self, x):
        swin_features = self.swin_backbone.forward_features(x)
        swin_logits = self.swin_backbone.head(swin_features)
        vig_features = self._vig_features(x)
        vig_logits = self.vig_head(vig_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        weights = torch.softmax(self.fusion_weights, dim=0)
        fused_features = (weights[0] * self.swin_projector(swin_features)
                          + weights[1] * self.vig_projector(vig_features))
        fusion_logits = self.fusion_classifier(fused_features)
        return swin_logits, vig_logits, fusion_logits

class Gated_Fusion(nn.Module):
    """Sample-adaptive gated feature fusion baseline for SwinT and ViG.

    A sigmoid gate generated from both projected embeddings decides, separately
    for every sample and feature dimension, how much to retain from each branch.
    """
    def __init__(self, swinT_base=None, vig_base=None, class_num=3, fusion_dim=512, dropout=0.2):
        super().__init__()
        if swinT_base is None or vig_base is None:
            raise ValueError('swinT_base and vig_base must both be provided.')

        self.swin_backbone = swinT_base
        self.vig_stem = vig_base.stem
        self.vig_pos_embed = vig_base.pos_embed
        self.vig_blocks = vig_base.backbone
        self.vig_num_blocks = vig_base.n_blocks
        self.vig_projection = vig_base.prediction[0:3]
        self.vig_head = vig_base.prediction[4]

        swin_dim = getattr(swinT_base, 'num_features', 768)
        self.swin_projector = nn.Sequential(nn.Linear(swin_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.vig_projector = nn.Sequential(nn.Linear(1024, fusion_dim), nn.LayerNorm(fusion_dim))
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.fusion_classifier = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, class_num),
        )
        for layer in (self.swin_projector[0], self.vig_projector[0],
                      self.gate_network[0], self.gate_network[3], self.fusion_classifier[2]):
            nn.init.trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)

    def _vig_features(self, x):
        x = self.vig_stem(x) + self.vig_pos_embed
        for index in range(self.vig_num_blocks):
            x = self.vig_blocks[index](x)
        x = F.adaptive_avg_pool2d(x, 1)
        return torch.flatten(self.vig_projection(x), 1)

    def forward(self, x, return_gate=False):
        swin_features = self.swin_backbone.forward_features(x)
        swin_logits = self.swin_backbone.head(swin_features)
        vig_features = self._vig_features(x)
        vig_logits = self.vig_head(vig_features.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        swin_projected = self.swin_projector(swin_features)
        vig_projected = self.vig_projector(vig_features)
        gate = self.gate_network(torch.cat((swin_projected, vig_projected), dim=1))
        fused_features = gate * swin_projected + (1.0 - gate) * vig_projected
        fusion_logits = self.fusion_classifier(fused_features)
        if return_gate:
            return swin_logits, vig_logits, fusion_logits, gate
        return swin_logits, vig_logits, fusion_logits

