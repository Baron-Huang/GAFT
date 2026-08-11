import torch
from torch import nn
import torch.nn.functional as F
from models.gcn import DeepGCN
class Res_Net(nn.Module):
	def __init__(self, class_num=3, base = None):
		super(Res_Net, self).__init__()
		self.layer1 = base.layer1
		self.layer2 = base.layer2
		self.layer3 = base.layer3
		self.layer4 = base.layer4
		self.maxpool = base.maxpool
		self.conv1 = base.conv1
		self.bn1 = base.bn1
		self.avgpool = base.avgpool
		self.relu = base.relu
		self.linear = nn.Linear(in_features=2048, out_features=class_num)
		self.top_flat = nn.Flatten(start_dim=1, end_dim=3)

	def forward(self, x):
		y = self.conv1(x)
		y = self.bn1(y)
		y = self.relu(y)
		y = self.maxpool(y)
		y = self.layer1(y)
		y = self.layer2(y)
		y = self.layer3(y)
		y = self.layer4(y)
		y = self.avgpool(y)
		y = self.top_flat(y)
		y = self.linear(y)
		return y


class Dense_Net(nn.Module):
	def __init__(self, class_num=None, base_net=None):
		super(Dense_Net, self).__init__()
		self.base_net = base_net.features
		self.top_AvgMp = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
		self.top_flat = nn.Flatten()
		self.top_linear_3 = nn.Linear(1024, class_num)
		#nn.init.kaiming_normal(self.top_linear_3.weight)

	def forward(self, x):
		y = self.base_net(x)
		y = self.top_AvgMp(y)
		y = self.top_flat(y)
		y = self.top_linear_3(y)
		return y


class ResNeXt_Net(nn.Module):
	def __init__(self, class_num=3, base = None):
		super(ResNeXt_Net, self).__init__()
		self.base = base
		self.linear = nn.Linear(in_features=2048, out_features=class_num)
		self.avrgp = nn.AvgPool2d(kernel_size=(7, 7))
		self.top_flat = nn.Flatten(start_dim=1, end_dim=3)

	def forward(self, x):
		y = self.base(x)
		y = y[0]
		y = self.avrgp(y)
		y = self.top_flat(y)
		y = self.linear(y)
		return y


class ViG_Net(nn.Module):
	def __init__(self, base_model=None):
		super(ViG_Net, self).__init__()
		self.stem = base_model.stem
		self.prediction = base_model.prediction
		self.block_0 = base_model.backbone[0]
		self.block_1 = base_model.backbone[1]
		self.block_2 = base_model.backbone[2]
		self.block_3 = base_model.backbone[3]
		self.block_4 = base_model.backbone[4]
		self.block_5 = base_model.backbone[5]
		self.block_6 = base_model.backbone[6]
		self.block_7 = base_model.backbone[7]
		self.block_8 = base_model.backbone[8]
		self.block_9 = base_model.backbone[9]
		self.block_10 = base_model.backbone[10]
		self.block_11 = base_model.backbone[11]

	def forward(self, x):
		y = self.stem(x)
		y = self.block_0(y)
		y = self.block_1(y)
		y = self.block_2(y)
		y = self.block_3(y)
		y = self.block_4(y)
		y = self.block_5(y)
		y = self.block_6(y)
		y = self.block_7(y)
		y = self.block_8(y)
		y = self.block_9(y)
		y = self.block_10(y)
		y = self.block_11(y)
		y = F.adaptive_avg_pool2d(y, 1)
		y = self.prediction(y)
		return y.squeeze(-1).squeeze(-1)


def ViG_ti_Net(gpu_device=0, class_num=3, **kwargs):
	class OptInit:
		def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
			self.k = num_knn  # neighbor num (default:9)
			self.conv = 'mr'  # graph conv layer {edge, mr}
			self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
			self.norm = 'batch'  # batch or instance normalization {batch, instance}
			self.bias = True  # bias of conv layer True or False
			self.n_blocks = 12  # number of basic blocks in the backbone
			self.n_filters = 192  # number of channels of deep features
			self.n_classes = class_num  # Dimension of out_channels
			self.dropout = drop_rate  # dropout rate
			self.use_dilation = True  # use dilated knn or not
			self.epsilon = 0.2  # stochastic epsilon for gcn
			self.use_stochastic = False  # stochastic for gcn, True or False
			self.drop_path = drop_path_rate

	opt = OptInit(**kwargs)
	ViG_ti_Net_base = DeepGCN(opt)
	return ViG_ti_Net_base

class SwinT_Net(nn.Module):
	def __init__(self, base_model=None, class_num = 3):
		super(SwinT_Net, self).__init__()
		self.layers_0 = base_model.layers[0]
		self.layers_1 = base_model.layers[1]
		self.layers_2 = base_model.layers[2]
		self.layers_3 = base_model.layers[3]
		self.patch_embed = base_model.patch_embed
		self.pos_drop = base_model.pos_drop
		self.norm = base_model.norm
		self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)
		self.head = base_model.head

	def forward(self, x):
		y = self.patch_embed(x)
		y = self.pos_drop(y)
		y = self.layers_0(y)
		y = self.layers_1(y)
		y = self.layers_2(y)
		y = self.layers_3(y)
		y = self.norm(y)
		y = y.permute(0, 2, 1)
		y = self.avgp(y)
		y = self.head(y.reshape(y.shape[0], y.shape[1]))
		return y

class MViT_Net(nn.Module):
	def __init__(self, class_num=3, base = None):
		super(MViT_Net, self).__init__()
		self.base = base
		self.linear = nn.Linear(in_features=768, out_features=class_num)
		self.avrgp = nn.AvgPool2d(kernel_size=(7, 7))
		self.top_flat = nn.Flatten(start_dim=1, end_dim=3)

	def forward(self, x):
		y = self.base(x)
		y = y[0]
		y = self.avrgp(y)
		y = self.top_flat(y)
		y = self.linear(y)
		return y

class T2T_ViT_Net(nn.Module):
	def __init__(self, class_num=3, base = None):
		super(T2T_ViT_Net, self).__init__()
		self.base = base
		self.linear = nn.Linear(in_features=384, out_features=class_num)
		self.top_avrgp = nn.AvgPool2d(kernel_size=(14, 14), stride=14)
		self.top_flat = nn.Flatten(start_dim=1, end_dim=3)

	def forward(self, x):
		y = self.base(x)
		y = y[0][0]
		y = self.top_avrgp(y)
		y = self.top_flat(y)
		y = self.linear(y)
		return y


class _NonLocalBlockND(nn.Module):
	def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
		super(_NonLocalBlockND, self).__init__()

		assert dimension in [1, 2, 3]

		self.dimension = dimension
		self.sub_sample = sub_sample

		self.in_channels = in_channels
		self.inter_channels = inter_channels

		if self.inter_channels is None:
			self.inter_channels = in_channels // 2
			if self.inter_channels == 0:
				self.inter_channels = 1

		if dimension == 3:
			conv_nd = nn.Conv3d
			max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
			bn = nn.BatchNorm3d
		elif dimension == 2:
			conv_nd = nn.Conv2d
			max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
			bn = nn.BatchNorm2d
		else:
			conv_nd = nn.Conv1d
			max_pool_layer = nn.MaxPool1d(kernel_size=(2))
			bn = nn.BatchNorm1d

		self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
						 kernel_size=1, stride=1, padding=0)

		if bn_layer:
			self.W = nn.Sequential(
				conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
						kernel_size=1, stride=1, padding=0),
				bn(self.in_channels)
			)
			nn.init.constant_(self.W[1].weight, 0)
			nn.init.constant_(self.W[1].bias, 0)
		else:
			self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
							 kernel_size=1, stride=1, padding=0)
			nn.init.constant_(self.W.weight, 0)
			nn.init.constant_(self.W.bias, 0)

		self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
							 kernel_size=1, stride=1, padding=0)

		self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
						   kernel_size=1, stride=1, padding=0)

		if sub_sample:
			self.g = nn.Sequential(self.g, max_pool_layer)
			self.phi = nn.Sequential(self.phi, max_pool_layer)

	def forward(self, x):
		'''
		:param x: (b, c, t, h, w)
		:return:
		'''

		batch_size = x.size(0)

		g_x = self.g(x).view(batch_size, self.inter_channels, -1)
		g_x = g_x.permute(0, 2, 1)

		theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
		theta_x = theta_x.permute(0, 2, 1)
		phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
		f = torch.matmul(theta_x, phi_x)
		N = f.size(-1)
		f_div_C = f / N

		y = torch.matmul(f_div_C, g_x)
		y = y.permute(0, 2, 1).contiguous()
		y = y.view(batch_size, self.inter_channels, *x.size()[2:])
		W_y = self.W(y)
		z = W_y + x

		return z

class NONLocalBlock2D(_NonLocalBlockND):
	def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
		super(NONLocalBlock2D, self).__init__(in_channels,
											  inter_channels=inter_channels,
											  dimension=2, sub_sample=sub_sample,
											  bn_layer=bn_layer)

class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y.expand_as(x)

class FABNet(nn.Module):
	def __init__(self, class_num=None, base_net=None):
		super(FABNet, self).__init__()
		#self.base_net = base_net.features
		self.input_module = base_net.features[0:4]
		self.dense_block_1 = base_net.features.denseblock1
		self.dense_block_2 = base_net.features.denseblock2
		self.dense_block_3 = base_net.features.denseblock3
		self.dense_block_4 = base_net.features.denseblock4
		self.transition_1 = base_net.features.transition1
		self.transition_2 = base_net.features.transition2
		self.transition_3 = base_net.features.transition3
		self.BN_5 = base_net.features.norm5

		self.top_AvgMp = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
		self.top_flat = nn.Flatten()
		#self.top_linear_1 = nn.Linear(1024 * 7 * 7, 1024)
		self.top_linear_3 = nn.Linear(1024, class_num)
		self.dp = nn.Dropout(p=0.3)
		self.SE_1 = SELayer(128, 8)
		self.SE_2 = SELayer(256, 16)
		self.SE_3 = SELayer(512, 24)
		self.SE_4 = SELayer(1024, 32)
		self.NonLocal_1 = NONLocalBlock2D(in_channels=128)
		self.NonLocal_2 = NONLocalBlock2D(in_channels=256)
		self.NonLocal_3 = NONLocalBlock2D(in_channels=512)
		self.NonLocal_4 = NONLocalBlock2D(in_channels=1024)
		#nn.init.kaiming_normal(self.top_linear_3.weight)

	def forward(self, x):
		#y = self.base_net(x)
		y = self.input_module(x)
		y = self.dense_block_1(y)
		y = self.transition_1(y)
		y_1 = self.SE_1(y)
		y_2 = self.NonLocal_1(y)
		y = (y_1 + y_2) / 2.0
		y = self.dense_block_2(y)
		y = self.transition_2(y)
		y_1 = self.SE_2(y)
		y_2 = self.NonLocal_2(y)
		y = (y_1 + y_2) / 2.0
		y = self.dense_block_3(y)
		y = self.transition_3(y)
		y_1 = self.SE_3(y)
		y_2 = self.NonLocal_3(y)
		y = (y_1 + y_2) / 2.0
		y = self.dense_block_4(y)
		y = self.BN_5(y)
		y_1 = self.SE_4(y)
		y_2 = self.NonLocal_4(y)
		y = (y_1 + y_2) / 2.0
		y = self.top_AvgMp(y)
		y = self.top_flat(y)
		#y = self.dp(y)
		#y = self.top_linear_1(y)
		#y = self.top_linear_2(y)
		y = self.top_linear_3(y)
		return y


class ViT_AMCNet(nn.Module):
	def __init__(self, vit_base = None, dense_base = None, work_mode = None, model_mode = None, class_num = 3):
		super(ViT_AMCNet, self).__init__()
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

		self.input_module = dense_base.features[0:4]
		self.dense_block_1 = dense_base.features.denseblock1
		self.dense_block_2 = dense_base.features.denseblock2
		self.dense_block_3 = dense_base.features.denseblock3
		self.dense_block_4 = dense_base.features.denseblock4
		self.transition_1 = dense_base.features.transition1
		self.transition_2 = dense_base.features.transition2
		self.transition_3 = dense_base.features.transition3
		self.BN_5 = dense_base.features.norm5
		self.top_AvgMp = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
		self.top_flat = nn.Flatten()
		self.top_linear_3 = nn.Linear(1024, class_num)
		self.dp = nn.Dropout(p=0.3)
		self.SE_1 = SELayer(128, 8)
		self.SE_2 = SELayer(256, 16)
		self.SE_3 = SELayer(512, 24)
		self.SE_4 = SELayer(1024, 32)
		self.NonLocal_1 = NONLocalBlock2D(in_channels=128)
		self.NonLocal_2 = NONLocalBlock2D(in_channels=256)
		self.NonLocal_3 = NONLocalBlock2D(in_channels=512)
		self.NonLocal_4 = NONLocalBlock2D(in_channels=1024)
		self.work_mode = work_mode
		self.model_mode = model_mode

		if self.model_mode == 'FDAF':
			self.fusion_block_dense_1 = nn.Linear(1792, 112)
			self.fusion_block_dense_2 = nn.Linear(112, 1792)
			self.fusion_layer = nn.Linear(1792, class_num)
		elif self.model_mode == 'FDAI' or self.model_mode =='MLMT' or self.model_mode =='Featrues_stacking':
			self.fusion_layer = nn.Linear(1792, class_num)

	def forward(self, x):
		vit_y = self.patch_embed(x)
		B = x.shape[0]
		cls_tokens = self.cls_token.expand(B, -1, -1)
		vit_y = torch.cat((cls_tokens, vit_y), dim=1)
		vit_y = vit_y + self.pos_embed
		vit_y = self.pos_drop(vit_y)
		vit_y = self.transformer_block_0(vit_y)
		vit_y = self.transformer_block_1(vit_y)
		vit_y = self.transformer_block_2(vit_y)
		vit_y = self.transformer_block_3(vit_y)
		vit_y = self.transformer_block_4(vit_y)
		vit_y = self.transformer_block_5(vit_y)
		vit_y = self.transformer_block_6(vit_y)
		vit_y = self.transformer_block_7(vit_y)
		vit_y = self.transformer_block_8(vit_y)
		vit_y = self.transformer_block_9(vit_y)
		vit_y = self.transformer_block_10(vit_y)
		vit_y = self.transformer_block_11(vit_y)
		vit_y_Fusion = self.norm(vit_y)
		vit_y = self.head(vit_y_Fusion)
		vit_y = vit_y[:, 0, :]

		y = self.input_module(x)
		y = self.dense_block_1(y)
		y = self.transition_1(y)
		y_1 = self.SE_1(y)
		y_2 = self.NonLocal_1(y)
		y = (y_1 + y_2) / 2.0
		y = self.dense_block_2(y)
		y = self.transition_2(y)
		y_1 = self.SE_2(y)
		y_2 = self.NonLocal_2(y)
		y = (y_1 + y_2) / 2.0
		y = self.dense_block_3(y)
		y = self.transition_3(y)
		y_1 = self.SE_3(y)
		y_2 = self.NonLocal_3(y)
		y = (y_1 + y_2) / 2.0
		y = self.dense_block_4(y)
		y = self.BN_5(y)
		y_1 = self.SE_4(y)
		y_2 = self.NonLocal_4(y)
		y = (y_1 + y_2) / 2.0
		y = self.top_AvgMp(y)
		y_Fusion = self.top_flat(y)
		fab_y = self.top_linear_3(y_Fusion)

		if self.model_mode == 'FDAF':
			###Adaptive Features Fusion
			fusion_out = torch.cat((y_Fusion, vit_y_Fusion[:, 0, :]), dim=1)
			fusion_weights = self.fusion_block_dense_1(fusion_out)
			fusion_weights = torch.nn.functional.relu(fusion_weights)
			fusion_weights = self.fusion_block_dense_2(fusion_weights)
			fusion_weights = torch.nn.functional.softmax(fusion_weights, dim=1)
			fusion_out = fusion_out * fusion_weights
			fusion_out = self.fusion_layer(fusion_out)
		elif self.model_mode == 'FDAI' or self.model_mode == 'MLMT' or self.model_mode == 'Features_stacking':
			###Features Fusion
			fusion_out = torch.cat((y_Fusion, vit_y_Fusion[:, 0, :]), dim=1)
			fusion_out = self.fusion_layer(fusion_out)
		elif self.model_mode == 'Probability_fusion':
			###Probability Fusion
			fusion_out = fab_y + vit_y


		if self.work_mode == 'features_extraction':
			return y_Fusion, vit_y_Fusion[:, 0, :], fusion_out
		elif self.work_mode == 'normal':
			return vit_y, fab_y, fusion_out


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

		vig_fusion_out = self.vig_fc1(fusion_out)
		vig_fusion_out = torch.nn.functional.relu(vig_fusion_out)
		vig_fusion_out = self.vig_fc2(vig_fusion_out)
		vig_fusion_out_weight = torch.nn.functional.sigmoid(vig_fusion_out)
		vig_fusion_out = vig_fusion_out_weight * vig_y_fusion

		fusion_out = torch.cat((swin_fusion_out, vig_fusion_out), dim=1)
		fusion_out = self.fusion_fc1(fusion_out)
		fusion_out = torch.nn.functional.relu(fusion_out)
		fusion_out = self.fusion_fc2(fusion_out)
		return swin_t_y, vig_y, fusion_out,swin_t_y_fusion, vig_y_fusion
