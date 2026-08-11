import torch
from torch import nn, no_grad
from torchvision.models.densenet import densenet121
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from GNN_cam_utils import ViG_CAM
from models.gcn import vig_ti_224_gelu
from models.swin_transformer import swin_tiny
from CAM_module import *
from utils.train_utils import train_single, train_baseline, train_moo_dis_lmmd, testing_funnction, CapturePrint, PrintCapture



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
	image_dir = r'/data/QYB/GMamba/dataset/Larynx_datasets_ST_MSLNet_Random'
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

	print('########################## models and testing models #########################')
	swin_base = swin_tiny(class_num=3)
	gcn_base = vig_ti_224_gelu(class_num=3)  # gat
	net = ViT2GNN_AAMFF_Dis(swinT_base=swin_base, vig_base=gcn_base)
	model_path = r'/data/QYB/ViT2GNN/result/lar/s/aamff_moo_dis_loss0.8*fusion_loss+0.8*swint_loss+0.8*vig_loss+0.25*dis_loss_4.pth'
	#model_path = r'/data/QYB/ViT2GNN/result/esophagel/all/aamff_moo_dis_loss1*fusion_loss+0.5*swint_loss+0.6*vig_loss+0.45*dis_loss.pth'

	out_mode = 'five'
	Grade = 'I'  # I, II, III
	datasets = 'lar'  # lar, eso
	with no_grad():
		net_weight = torch.load(model_path)
		net.load_state_dict(net_weight, strict=True)
	net = net.cuda(gpu_device)
	testing_funnction(test_model=net, train_loader=train_loader, val_loader=val_loader,
					  test_loader=test_loader, gpu_device=gpu_device, out_mode=out_mode)
	print('########################## Visually analyze #########################')
	path_name = image_dir + r'/Test_patch/'+str(Grade)+''
	img_name_list = os.listdir(path_name)
	result_line_img_path = r'/data/QYB/ViT2GNN/gnn_picture/eso/'+str(Grade)+''
	result_feature_map_path = r'E:\QYB\ViG_SwinT\results\VIG_out\feature\I'
	block_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	for img_name in img_name_list:
		for idx, i in enumerate(block_select):
			my_cam = ViG_CAM(model=net, out_model='five', transform=transform, i = i,
						 path_name=path_name, img_name=img_name,
						 result_line_img_path=result_line_img_path,
						 result_feature_map_path=result_feature_map_path, gap=1)
			my_cam.get_all_result_images()







