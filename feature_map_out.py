###swin & swinmlp
import PIL
import torch
from torch import nn
import seaborn as sns
import numpy as np
from skimage import io
from torchvision import transforms
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
sns.set(font='Times New Roman', font_scale=0.6)

class LayerOutputHook:
	def __init__(self):
		self.output = None

	def __call__(self, module, input, output):
		self.output = output


class feature_out:
	def __init__(self, model = None, model_name = None,
				 out_model=None, i = None, path_name = None,
				 transform = None, gpu_device = 0,
				img_name = None,
				result_feature_map_path = None):
		self.i = i
		self.model = model
		self.model_name = model_name
		self.out_model = out_model
		self.path_name = path_name
		self.img_name = img_name
		self.transform = transform
		self.gpu_device = gpu_device
		self.result_feature_map_path = result_feature_map_path

	def bese_function(self):
		original_img = io.imread(self.path_name + r'/' + self.img_name)  #原图
		vis_img = PIL.Image.open(self.path_name + r'/' + self.img_name)
		transform_img = transforms.Resize([224, 224])
		vis_show_img = transform_img(vis_img)   #224*224 图
		vis_img_tr = self.transform(vis_img)    #输入进网络的图片
		vis_img_tr = vis_img_tr.view(1, 3, 224, 224)
		self.model.eval()
		self.model.zero_grad()
		if self.i == 0:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_0.blocks[0].mlp.fc2    #1,3196,96
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[0].blocks[0].mlp.fc2   #1,3168,128
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer1[1].conv3   #56,56
		if self.i == 1:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_0.blocks[1].mlp.fc2    #1,3196,96
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[0].blocks[1].mlp.fc2   #1,784,256
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer1[2].conv3  #56.56
		if self.i == 2:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_1.blocks[0].mlp.fc2    #1,784,192
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[1].blocks[0].mlp.fc2   #1,196,512
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer2[2].conv3 #28 28
		if self.i == 3:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_1.blocks[1].mlp.fc2
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[1].blocks[1].mlp.fc2
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer2[3].conv3 #28 28
		if self.i == 4:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_2.blocks[0].mlp.fc2    #1,196,384
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[2].blocks[0].mlp.fc2
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer3[0].conv3 #14 14
		if self.i == 5:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_2.blocks[1].mlp.fc2
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[2].blocks[3].mlp.fc2
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer3[1].conv3 #14 14
		if self.i == 6:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_2.blocks[2].mlp.fc2
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[2].blocks[6].mlp.fc2
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer3[2].conv3 #14 14
		if self.i == 7:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_2.blocks[3].mlp.fc2
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[2].blocks[9].mlp.fc2
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer3[3].conv3 #14 14
		if self.i == 8:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_2.blocks[4].mlp.fc2
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[2].blocks[12].mlp.fc2
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer3[4].conv3 #14 14
		if self.i == 9:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_2.blocks[5].mlp.fc2
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[2].blocks[15].mlp.fc2
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer3[5].conv3 #14 14
		if self.i == 10:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_3.blocks[0].mlp.fc2
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[3].blocks[0].mlp.fc2
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer4[1].conv3 #7 7
		if self.i == 11:
			if self.model_name == 'swint':
				feature_layer = self.model.layers_3.blocks[1].mlp.fc2    #1,49,768
			elif self.model_name == 'swinmlp':
				feature_layer = self.model.layers[3].blocks[1].mlp.fc2
			elif self.model_name == 'resnext':
				feature_layer = self.model.base.layer4[2].conv3 #7 7
		hook_feature_layer = LayerOutputHook()
		hook_target_layer_handle = feature_layer.register_forward_hook(hook_feature_layer)


		if self.out_model == 'single':
			pre_y = self.model(vis_img_tr.cuda(self.gpu_device))  # 预测矩阵
		if self.out_model == 'triplet':
			_, _, pre_y = self.model(vis_img_tr.cuda(self.gpu_device))  # 预测矩阵
		target_layer_out = hook_feature_layer.output                  #目标层选点

		hook_target_layer_handle.remove()

		pre_cls_num = torch.argmax(nn.Softmax(dim=-1)(pre_y)).detach().cpu().numpy()   #预测类别
		pre_proba = nn.Softmax(dim=-1)(pre_y).detach().cpu().numpy()
		pre_proba = pre_proba[0, pre_cls_num]                                    #预测概率值

		feature_map = feature_graph_out(img_name=self.img_name, target_layer_out=target_layer_out,
										result_feature_map_path=self.result_feature_map_path,
										block_num=self.i, model_name=self.model_name)
		# 分割效果图，大图分割后的小图位置索引


		return (original_img,  feature_map, pre_proba, pre_cls_num)


def feature_graph_out(img_name = None,
					target_layer_out: object = None,
					block_num = None,
					result_feature_map_path = None,
					model_name = None,
					 ) -> object:
#############################################从featuremap进行选点################################################
	if model_name == 'swint' or model_name == 'swinmlp':
		A, B, C = target_layer_out.shape  # 1, 3168 ,128   1,49,768
		D = int(B ** 0.5)
		target_layer_out = target_layer_out.view(A, D, D, C)
		point_tensor, _ = torch.max(target_layer_out, dim=3, keepdim=True)
	elif model_name == 'resnext':
		point_tensor, _ = torch.max(target_layer_out, dim=1, keepdim=True)
	point_tensor = point_tensor.squeeze()
	point_tensor_cpu = point_tensor.cpu()
	feature_map = np.array(point_tensor_cpu.detach().numpy())
	feature_map = plt.imshow(feature_map, cmap='viridis')
	plt.colorbar(feature_map, ticks=[])
	plt.axis('off')

	base_name = os.path.splitext(img_name)[0]
	img_name = f"{base_name}_{block_num}.jpg"
	print(img_name)
	feature_map_path = result_feature_map_path + r'/' + img_name

	plt.savefig(feature_map_path)
	plt.show()
