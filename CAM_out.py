#ResNet50, DenseNet121, ResNext, ViG, Swin_T, MViT, T2T-ViT, FABNeT, ViT_AMC, Mine
import torch
from pyarrow.dataset import dataset
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import random
from torchsummary import summary
from SOTA.SIL_Utils.fit_functions import vit_lr_schedule, Single_out_fit,searching_best_lr, testing_funnction
import seaborn as sns
from SOTA.SIL_Learning.Adaptive_Dual_Semantic_CAM import Adaptive_Dual_Semantic_CAM
from SOTA.SIL_Learning.CAM import CAM
from SOTA.SIL_Learning.Grad_CAM import Grad_CAM
#from SOTA.SIL_Learning.Grad_CAM_Plus2 import Grad_CAM_Plus2
#from SOTA.SIL_Learning.Score_CAM import Score_CAM
#from SOTA.SIL_Learning.Smooth_Grad_CAM_PP import Smooth_Grad_CAM_Plus2
#from SOTA.SIL_Learning.Layer_CAM import Layer_CAM
from SOTA.SIL_Learning.XGrad_CAM import XGrad_CAM
from CAM_module import *
from torchvision.models.resnet import resnet50
from torchvision.models.densenet import densenet121
from mmcls.models.backbones.resnext import ResNeXt
from models.gcn import vig_ti_224_gelu
from models.swin_transformer import swin_tiny
from mmcls.models.backbones.mvit import MViT
from mmcls.models.backbones.t2t_vit import T2T_ViT
from SOTA.SIL_Model.ViT_models.ViT import VisionTransformer


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	np.random.seed(seed)
	random.seed(seed)

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

	print(f"Random seed set as {seed}")

########################## main_function #########################
if __name__ == '__main__':
	setup_seed(0)
	gpu_device = 0
	class_num = 3
	batch_size = 32
	epochs = 100
	image_dir = r'/data/QYB/GMamba/dataset/breast_dataset'

	model_name = 'Swin_T'  # ResNet50, DenseNet121, ResNext, ViG, Swin_T, MViT, T2T_ViT, FABNeT, ViT_AMC, Mine
	#Mine use its py --- train_moo_AAMFF_dis
	Grade = 'I'  #  I, II, III
	datasets = 'bre' #  lar, eso, bre

	print('########################## reading datas and processing datas #########################')
	transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224, 224]),
									transforms.Normalize(mean=0.5, std=0.5)])
	train_data = ImageFolder(image_dir + r'/Train_patch', transform=transform)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
	val_data = ImageFolder(image_dir + r'/Val_patch', transform=transform)
	val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1)
	test_data = ImageFolder(image_dir + r'/Test_patch', transform=transform)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)
	print(train_data,'/n')
	print('########################## select model #########################')

	if model_name == 'ResNet50':
		res_base = resnet50()
		net = Res_Net(base=res_base, class_num=class_num)
		model_path = r'/data/QYB/GMamba/result/contrast/ResNet50/Resnet50_'+str(datasets)+'.pth'
		out_mode = 'single'
	elif model_name == 'DenseNet121':
		dense_base = densenet121()
		net = Dense_Net(base_net=dense_base, class_num=class_num)
		model_path = r'/data/QYB/GMamba/result/contrast/DenseNet121/Dense121_'+str(datasets)+'.pth'
		out_mode = 'single'
	elif model_name == 'ResNext':
		resnsxt_base = ResNeXt(depth = 50)
		net = ResNeXt_Net(base=resnsxt_base, class_num=class_num)
		model_path = r'/data/QYB/GMamba/result/contrast/ResNeXt50/resnext_'+str(datasets)+'.pth'
		out_mode = 'single'
	elif model_name == 'ViG':
		net_ = ViG_ti_Net(class_num=3)
		net = ViG_Net(base_model=net_)
		model_path = r'/data/QYB/GMamba/result/contrast/ViG_ti/ViG_ti_'+str(datasets)+'.pth'
		out_mode = 'single'
	elif model_name == 'Swin_T':
		swin_base = swin_tiny(class_num=class_num)
		net = SwinT_Net(base_model=swin_base,class_num=class_num)
		model_path = r'/data/QYB/GMamba/result/contrast/Swin_ti/SwinT_'+str(datasets)+'.pth'
		out_mode = 'single'
	elif model_name == 'MViT':
		mvit_base = MViT(arch='base')
		net = MViT_Net(base=mvit_base,class_num=class_num)
		model_path = r'/data/QYB/GMamba/result/contrast/MViTV2_base_1k/mvit_'+str(datasets)+'.pth'
		out_mode = 'single'
	elif model_name == 'T2T_ViT':
		t2t_vit_base = T2T_ViT()
		net = T2T_ViT_Net(base=t2t_vit_base,class_num=class_num)
		model_path = r'/data/QYB/GMamba/result/contrast/T2T_ViT_t_1k/t2t_vit_'+str(datasets)+'.pth'
		out_mode = 'single'
	elif model_name == 'FABNeT':
		dense_base = densenet121(pretrained=False)
		net = FABNet(base_net=dense_base, class_num=class_num)
		model_path = r'/data/QYB/GMamba/result/contrast/FABNet_change_by_densenet121/FABNet_'+str(datasets)+'.pth'
		out_mode = 'single'
	elif model_name == 'ViT_AMC':
		vit_base = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=class_num, embed_dim=768,
									 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
									 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
									 norm_layer=nn.LayerNorm)
		dense_base = densenet121(pretrained=False)
		net = ViT_AMCNet(vit_base=vit_base, dense_base=dense_base, work_mode='normal', model_mode='FDAF', class_num=class_num)
		model_path = r'/data/QYB/GMamba/result/contrast/ViT_AMC/ViT_AMC_'+str(datasets)+'.pth'
		print(model_path)
		out_mode = 'triplet'
	elif model_name == 'Mine':
		swin_base = swin_tiny(class_num=3)
		gcn_base = vig_ti_224_gelu(class_num=3)  # gat
		net = ViT2GNN_AAMFF_Dis(swinT_base=swin_base, vig_base=gcn_base)
		model_path = r'/data/QYB/ViT2GNN/result/esophagel/all/aamff_moo_dis_loss1*fusion_loss+0.5*swint_loss+0.6*vig_loss+0.45*dis_loss.pth'
		out_mode = 'five'

	# with torch.no_grad():
	# 	summary(model=net, input_size=(3, 224, 224), device='cpu')
	# 	print(net)
	net_weight = torch.load(model_path)
	net.load_state_dict(net_weight, strict=True)
	net = net.cuda(gpu_device)



	print('########################## testing function #########################')
	#testing_funnction(test_model=net,train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,gpu_device=gpu_device,out_mode = out_mode)

	path_name = image_dir + r'/Test_patch/'+str(Grade)+''
	img_name_list = os.listdir(path_name)
	result_img_path = r'/data/QYB/SwinGAT/cam_picture/' + (datasets) + '/' + (model_name) + '/' + (Grade) + '' #ViT2GNN
	result_cam_path = r'/data/QYB/SwinGAT/cam_picture/' + (datasets) + '/' + (model_name) + '/' + (Grade) + ''

	for i in img_name_list:
		my_grad_cam = Grad_CAM(model = net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
			   inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True, model_name=model_name,out_mode=out_mode,
				training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

		my_grad_cam.get_all_result_images()

		# my_grad_cam_pp = Grad_CAM_Plus2(model = swinT_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
		#        inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
		#         training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

		# my_score_cam = Score_CAM(model = swinT_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
		#        inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
		#         training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

		# my_s_grad_cam_pp = Smooth_Grad_CAM_Plus2(model = swinT_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
		#        inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
		#         training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

		# my_layer_cam = Layer_CAM(model=swinT_net, path_name=path_name, img_name=i,
		#                result_cam_path=result_cam_path,inverse_set=True, transform=transform, gpu_device=gpu_device,
		#                show_all_fp=True, training_img_path=None, result_img_path=result_img_path, map_size=[7, 7])

		# my_xgrad_cam = XGrad_CAM(model=net, path_name=path_name, img_name=i, result_cam_path=result_cam_path,
		#                          inverse_set=True, transform=transform, gpu_device=gpu_device, show_all_fp=True,
		#                          training_img_path=None, result_img_path=result_img_path, map_size=[7, 7])

		# my_ads_cam = Adaptive_Dual_Semantic_CAM(model = swinT_net, path_name = path_name, img_name = i,
		#                                        select_cls = 2, inverse_set = True, set_rate_1 = 0.65,
		#                                        show_all_fp = True, set_rate_2 = 0.65, rd_setting = True,
		#                                        transform=transform, training_img_path = training_img_path,
		#                                        ads_cam_model = None, result_img_path = result_img_path,
		#                                        result_cam_path = result_cam_path, out_mode= 'triplet',
		#                                        adaptive_strategy = 'histopathology')

		# my_cam = CAM(model = swinT_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
		#          select_cls = 0, inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
		#         training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

