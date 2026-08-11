import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_


class ViT2GNN(nn.Module):
	def __init__(self, vit_base=None, vig_base=None):
		super(ViT2GNN, self).__init__()
		###############ViT#################
		self.patch_embed = vit_base.patch_embed
		self.cls_token = vit_base.cls_token
		self.pos_embed = vit_base.pos_embed
		self.pos_drop = vit_base.pos_drop
		self.transformer_block_0 = vit_base.blocks[0]
		self.transformer_block_1 = vit_base.blocks[1]
		self.transformer_block_2 = vit_base.blocks[2]
		self.transformer_block_3 = vit_base.blocks[3]
		self.transformer_block_4 = vit_base.blocks[4]
		self.transformer_block_5 = vit_base.blocks[5]
		self.transformer_block_6 = vit_base.blocks[6]
		self.transformer_block_7 = vit_base.blocks[7]
		self.transformer_block_8 = vit_base.blocks[8]
		self.transformer_block_9 = vit_base.blocks[9]
		self.transformer_block_10 = vit_base.blocks[10]
		self.transformer_block_11 = vit_base.blocks[11]
		self.norm = vit_base.norm
		self.head = vit_base.head
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
		############################ViT stage0##########################
		vit_y = self.patch_embed(x)
		B = x.shape[0]
		cls_token = self.cls_token.expand(B, -1, -1)
		vit_y = torch.cat((vit_y, cls_token), dim=1)
		vit_y = vit_y + self.pos_embed
		vit_y = self.pos_drop(vit_y)
		vit_y = self.transformer_block_0(vit_y)
		vit_y_0 = self.transformer_block_1(vit_y)

		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################ViT stage1##########################
		vit_y = self.transformer_block_2(vit_y_0)
		vit_y_1 = self.transformer_block_3(vit_y)

		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################ViT stage2##########################
		vit_y = self.transformer_block_4(vit_y_1)
		vit_y = self.transformer_block_5(vit_y)
		vit_y = self.transformer_block_6(vit_y)
		vit_y = self.transformer_block_7(vit_y)
		vit_y = self.transformer_block_8(vit_y)
		vit_y_2 = self.transformer_block_9(vit_y)

		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################ViT stage3##########################
		vit_y = self.transformer_block_10(vit_y_2)
		vit_y_3 = self.transformer_block_11(vit_y)

		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		vit_y_fusion = self.norm(vit_y_3)
		vit_y = self.head(vit_y_fusion)
		vit_y = vit_y[:,0,:]
		fusion_out = vit_y + vig_y

		return vit_y, vig_y, fusion_out


class ViT2GNN_FDAF(nn.Module):
	def __init__(self, vit_base=None, vig_base=None):
		super(ViT2GNN_FDAF, self).__init__()
		###############ViT#################
		self.patch_embed = vit_base.patch_embed
		self.cls_token = vit_base.cls_token
		self.pos_embed = vit_base.pos_embed
		self.pos_drop = vit_base.pos_drop
		self.transformer_block_0 = vit_base.blocks[0]
		self.transformer_block_1 = vit_base.blocks[1]
		self.transformer_block_2 = vit_base.blocks[2]
		self.transformer_block_3 = vit_base.blocks[3]
		self.transformer_block_4 = vit_base.blocks[4]
		self.transformer_block_5 = vit_base.blocks[5]
		self.transformer_block_6 = vit_base.blocks[6]
		self.transformer_block_7 = vit_base.blocks[7]
		self.transformer_block_8 = vit_base.blocks[8]
		self.transformer_block_9 = vit_base.blocks[9]
		self.transformer_block_10 = vit_base.blocks[10]
		self.transformer_block_11 = vit_base.blocks[11]
		self.norm = vit_base.norm
		self.head = vit_base.head
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

		self.fusion_block_dense_1 = nn.Linear(1792, 224)
		self.fusion_block_dense_2 = nn.Linear(224, 1792)
		self.fusion_layer = nn.Linear(1792, 3)


	def forward(self, x):
		#############################ViG stage0##########################
		vig_y = self.stem(x)
		vig_y = self.block_0(vig_y)
		vig_y_0 = self.block_1(vig_y)  # 16，192，14，14  stage0 out
		############################ViT stage0##########################
		vit_y = self.patch_embed(x)
		B = x.shape[0]
		cls_token = self.cls_token.expand(B, -1, -1)
		vit_y = torch.cat((vit_y, cls_token), dim=1)
		vit_y = vit_y + self.pos_embed
		vit_y = self.pos_drop(vit_y)
		vit_y = self.transformer_block_0(vit_y)
		vit_y_0 = self.transformer_block_1(vit_y)

		#############################ViG stage1##########################
		vig_y = self.block_2(vig_y_0)
		vig_y_1 = self.block_3(vig_y)  # 16，192，14，14  stage1 out
		#############################ViT stage1##########################
		vit_y = self.transformer_block_2(vit_y_0)
		vit_y_1 = self.transformer_block_3(vit_y)

		#############################ViG stage2##########################
		vig_y = self.block_4(vig_y_1)
		vig_y = self.block_5(vig_y)
		vig_y = self.block_6(vig_y)
		vig_y = self.block_7(vig_y)
		vig_y = self.block_8(vig_y)
		vig_y_2 = self.block_9(vig_y)  # 16，192，14，14  stage2 out
		#############################ViT stage2##########################
		vit_y = self.transformer_block_4(vit_y_1)
		vit_y = self.transformer_block_5(vit_y)
		vit_y = self.transformer_block_6(vit_y)
		vit_y = self.transformer_block_7(vit_y)
		vit_y = self.transformer_block_8(vit_y)
		vit_y_2 = self.transformer_block_9(vit_y)

		#############################ViG stage3##########################
		vig_y = self.block_10(vig_y_2)
		vig_y_3 = self.block_11(vig_y)  # 16，192，14，14  stage3 out
		#############################ViT stage3##########################
		vit_y = self.transformer_block_10(vit_y_2)
		vit_y_3 = self.transformer_block_11(vit_y)

		#######################ViG fusion&out part#######################
		vig_y = F.adaptive_avg_pool2d(vig_y_3, 1)
		vig_y = self.prediction(vig_y)
		vig_y_fusion = self.flat(vig_y)
		vig_y = self.vig_out(vig_y).squeeze(-1).squeeze(-1)
		#####################SwinT fusion&out part#######################
		vit_y_fusion = self.norm(vit_y_3)
		vit_y = self.head(vit_y_fusion)
		vit_y = vit_y[:,0,:]

		fusion_out = torch.cat((vig_y_fusion, vit_y_fusion[:,0,:]), dim=1)
		fusion_weights = self.fusion_block_dense_1(fusion_out)
		fusion_weights = torch.nn.functional.relu(fusion_weights)
		fusion_weights = self.fusion_block_dense_2(fusion_weights)
		fusion_weights = torch.nn.functional.softmax(fusion_weights, dim=1)
		fusion_out = fusion_out * fusion_weights
		fusion_out = self.fusion_layer(fusion_out)


		return vit_y, vig_y, fusion_out