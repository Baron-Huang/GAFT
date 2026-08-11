import torch
import io
import contextlib
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import random
from utils.train_utils import train_single, train_baseline, train_moo_aamff, testing_funnction, CapturePrint, PrintCapture
from models.swin_transformer import swin_tiny
from models.gcn import vig_ti_224_gelu
from models.my_net import ViT2GNN, ViT2GNN_AAMFF,ViT2GNN_AAMFF_Dis
from picture_out.jianyan import features_extraction_without_classes, analyze_features

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	np.random.seed(seed)
	random.seed(seed)

	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.use_deterministic_algorithms(True)

	print(f"Random seed set as {seed}")


def worker_init_fn(worker_id):
	random.seed(7 + worker_id)
	np.random.seed(7 + worker_id)
	torch.manual_seed(7 + worker_id)
	torch.cuda.manual_seed(7 + worker_id)
	torch.cuda.manual_seed_all(7 + worker_id)

if __name__ == '__main__':
	setup_seed(0)
	class_num = 3
	batch_size = 32
	epochs = 100
	image_size = 224
	num_workers = 1
	data_root = r'./dataset/esophageal_cancer'
	swinT_path = r'./weight/swin_tiny_patch4_window7_224_22k.pth'
	gcn_path = r'./weight/vig_ti_74.5.pth'
	device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
	print('########################## reading datas and processing datas #########################')
	transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([image_size, image_size], antialias=True),
									transforms.Normalize(mean=0.5, std=0.5)])
	train_data = ImageFolder(data_root + r'/Train_patch', transform=transform)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_data = ImageFolder(data_root + r'/Val_patch', transform=transform)
	val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_data = ImageFolder(data_root + r'/Test_patch', transform=transform)
	test_loader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=num_workers)

	for i in range(1):
		###########################################Swin_Part################################################
		swin_base = swin_tiny(class_num=class_num)
		weight_file = torch.load(swinT_path, map_location='cuda:1')["model"]
		weight_file = {k: v for k, v in weight_file.items() if (k in weight_file and 'head' not in k)}
		swin_base.load_state_dict(weight_file, strict=False)
		nn.init.trunc_normal_(swin_base.head.weight, std=.02)
		###########################################GCN_Part################################################
		gcn_base = vig_ti_224_gelu(class_num=class_num) #gat
		weight_file = torch.load(gcn_path, map_location='cuda:1')
		weight_file = {k: v for k, v in weight_file.items() if (k in weight_file and 'prediction.4' not in k)}
		gcn_base.load_state_dict(weight_file, strict=False)
		nn.init.trunc_normal_(gcn_base.prediction[4].weight, std=.02)


		with torch.no_grad():
			fusion_net_before = ViT2GNN_AAMFF(swinT_base=swin_base, vig_base=gcn_base)
		fusion_net_before = fusion_net_before.to(device)
		fusion_net_before_path = r'/data/QYB/ViT2GNN/result/esophagel/all/aamff_moo_1*fusion_loss+0.5*swint_loss+0.6*vig_loss.pth'
		fusion_net_before.load_state_dict(torch.load(fusion_net_before_path, map_location='cuda:1'))
		# testing_funnction(test_model=fusion_net_before, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
		# 								  gpu_device=device, out_mode='five')
		#features_swin, features_vig, labels = features_extraction_without_classes(test_model=fusion_net_before,
																  # data_loader=test_loader, gpu_device=1)
		#analyze_features(features_swin, features_vig, labels, output_dir=r'/data/QYB/ViT2GNN/')

		torch.cuda.empty_cache()

		with torch.no_grad():
			fusion_net_after = ViT2GNN_AAMFF_Dis(swinT_base=swin_base, vig_base=gcn_base)
		fusion_net_after = fusion_net_after.to(device)
		fusion_net_after_path = r'/data/QYB/ViT2GNN/result/esophagel/all/aamff_moo_dis_loss1*fusion_loss+0.5*swint_loss+0.6*vig_loss+0.45*dis_loss.pth'
		eso_weight = torch.load(fusion_net_after_path)
		fusion_net_after.load_state_dict(eso_weight, strict=True)
		# testing_funnction(test_model=fusion_net_after, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
		# 		  gpu_device=device, out_mode='five')
		features_swin, features_vig, labels = features_extraction_without_classes(test_model=fusion_net_after, data_loader=test_loader, gpu_device=1)

	# 执行分析流程
		analyze_features(features_swin, features_vig, labels, output_dir=r'/data/QYB/ViT2GNN/')
