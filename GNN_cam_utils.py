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


class ViG_CAM:
    def __init__(self, model = None, out_model=None, i = None, path_name = None,
                 transform = None, gpu_device = 0, show_all_fp = True,
                 training_img_path = None, img_name = None, gap = None,
                 result_line_img_path = None, result_feature_map_path = None):
        self.i = i
        self.gap = gap
        self.model = model
        self.out_model = out_model
        self.path_name = path_name
        self.img_name = img_name
        self.transform = transform
        self.gpu_device = gpu_device
        self.show_all_fp = show_all_fp
        self.training_img_path = training_img_path
        self.result_line_img_path = result_line_img_path
        self.result_feature_map_path = result_feature_map_path

    def bese_function(self):
        original_img = io.imread(self.path_name + r'/' + self.img_name)  #原图
        vis_img = PIL.Image.open(self.path_name + r'/' + self.img_name)
        transform_img = transforms.Resize([224, 224])
        vis_show_img = transform_img(vis_img)   #224*224 图
        vis_img_tr = self.transform(vis_img)    #输入进网络的图片
        vis_img_tr = vis_img_tr.view(1,3,224,224)
        self.model.eval()
        self.model.zero_grad()
        target_layer = None
        point_tagert_layer = None
        if self.i == 0:
            target_layer = self.model.block_0[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_0[0].fc1[1]
        if self.i == 1:
            target_layer = self.model.block_1[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_1[0].fc1[1]
        if self.i == 2:
            target_layer = self.model.block_2[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_2[0].fc1[1]
        if self.i == 3:
            target_layer = self.model.block_3[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_3[0].fc1[1]
        if self.i == 4:
            target_layer = self.model.block_4[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_4[0].fc1[1]
        if self.i == 5:
            target_layer = self.model.block_5[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_5[0].fc1[1]
        if self.i == 6:
            target_layer = self.model.block_6[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_6[0].fc1[1]
        if self.i == 7:
            target_layer = self.model.block_7[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_7[0].fc1[1]
        if self.i == 8:
            target_layer = self.model.block_8[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_8[0].fc1[1]
        if self.i == 9:
            target_layer = self.model.block_9[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_9[0].fc1[1]
        if self.i == 10:
            target_layer = self.model.block_10[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_10[0].fc1[1]
        if self.i == 11:
            target_layer = self.model.block_11[0].graph_conv.dilated_knn_graph._dilated
            point_tagert_layer = self.model.block_11[0].fc1[1]

        hook_target_layer = LayerOutputHook()
        hook_point_tagert_layer = LayerOutputHook()
        hook_target_layer_handle = target_layer.register_forward_hook(hook_target_layer)
        hook_point_tagert_layer_handle = point_tagert_layer.register_forward_hook(hook_point_tagert_layer)


        if self.out_model == 'single':
            pre_y = self.model(vis_img_tr.cuda(self.gpu_device))  # 预测矩阵
        if self.out_model == 'triplet':
            _, _, pre_y = self.model(vis_img_tr.cuda(self.gpu_device))  # 预测矩阵
        if self.out_model == 'five':
            _, _, pre_y, _, _ = self.model(vis_img_tr.cuda(self.gpu_device))  # 预测矩阵
        target_layer_idx = hook_target_layer.output                  #目标层选点
        point_tagert_layer_idx = hook_point_tagert_layer.output      #1,192,14,14

        hook_target_layer_handle.remove()
        hook_point_tagert_layer_handle.remove()

        pre_cls_num = torch.argmax(nn.Softmax(dim=-1)(pre_y)).detach().cpu().numpy()   #预测类别
        pre_proba = nn.Softmax(dim=-1)(pre_y).detach().cpu().numpy()
        pre_proba = pre_proba[0, pre_cls_num]                                    #预测概率值

        cutting_img, feature_map, line_image\
        = point_graph_out(image=vis_show_img,
                          img_name = self.img_name,
                          target_layer_idx = target_layer_idx,
                          point_tagert_layer_idx = point_tagert_layer_idx,
                          result_feature_map_path = self.result_feature_map_path,
                          block_num = self.i,
                          img_gap =self.gap)
        # 分割效果图，大图分割后的小图位置索引

        return (original_img, cutting_img, feature_map, target_layer_idx, point_tagert_layer_idx,
                line_image, pre_proba, pre_cls_num)

    ########################## get_result_images #########################
    def get_all_result_images(self):
        (original_img, cutting_img, feature_map, target_layer_idx, point_tagert_layer_idx,
         line_image, pre_proba, pre_cls_num)\
            = self.bese_function()

        base_name = os.path.splitext(self.img_name)[0]
        img_name = f"{base_name}_{self.i}.jpg"

        #feature_map_data = feature_map.get_array()
        #plt.imsave(self.result_feature_map_path + r'\\' + img_name, feature_map_data, cmap='magma')
        line_image.save(self.result_line_img_path + r'/' + img_name)

        print(self.img_name)

import torch
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import math


def point_graph_out(image: object = None,
					img_name = None,
					target_layer_idx = None,
					point_tagert_layer_idx: object = None,
					block_num = None,
					result_feature_map_path = None,
					 img_gap: object = 1) -> object:
	#########################################################裁切图片#######################################################
	width, height = image.size
	small_width = width // 14
	small_height = height // 14
	visual_width = width + 13 * img_gap
	visual_height = height + 13 * img_gap
	visual_image = Image.new("RGB", (visual_width, visual_height), "white")    #创建白色底图
	draw = ImageDraw.Draw(visual_image)
	small_images_with_index = []                    #小图片列表
	index = 0
	for i in range(14):
		for j in range(14):
			left = j * small_width
			upper = i * small_height
			right = left + small_width
			lower = upper + small_height
			small_image = image.crop((left, upper, right, lower))
			visual_left = j * (small_width + img_gap)
			visual_upper = i * (small_height + img_gap)
			# 将小图片粘贴到可视化图像上
			visual_image.paste(small_image, (visual_left, visual_upper))
			# 将小图片的位置索引添加到列表中
			small_images_with_index.append((small_image, index))
			index += 1
	small_images_with_index = np.asarray(small_images_with_index, dtype=object)
	visual_image_out = Image.new(visual_image.mode, visual_image.size)
	visual_image_out.paste(visual_image)                                  #visual_image_out输出分开小图的效果图
	plt.figure(1)
	plt.imshow(visual_image_out)
	plt.axis('off')
	#plt.show()
	#############################################从featuremap进行选点################################################
	point_tensor, _ = torch.max(point_tagert_layer_idx, dim=1, keepdim=True)
	point_tensor = point_tensor.squeeze()
	point_tensor_cpu = point_tensor.cpu()
	feature_map = np.array(point_tensor_cpu.detach().numpy())
	feature_map = plt.imshow(feature_map, cmap='magma')
	plt.colorbar(feature_map, ticks=[])
	plt.axis('off')

	base_name = os.path.splitext(img_name)[0]
	img_name = f"{base_name}_{block_num}.jpg"
	feature_map_path = result_feature_map_path + r'\\' + img_name

	#plt.savefig(feature_map_path)
	plt.show()
	# 将张量展平为一维
	flattened_point_tensor = point_tensor_cpu.view(-1)
	# 获取排序后的索引
	sorted_indices = torch.argsort(flattened_point_tensor)
	# 获取最大值和最小值的位置索引
	max_index = sorted_indices[-1]
	min_index = sorted_indices[0]
	# 获取最大值和最小值
	max_value = flattened_point_tensor[max_index]
	min_value = flattened_point_tensor[min_index]
	print(f"block:{block_num+1}, Max value: {max_value.item()}, Index: {max_index.item()}")
	print(f"block:{block_num+1}, Min value: {min_value.item()}, Index: {min_index.item()}")
	###################################第一个点（featuremap max） #########################
	dot_radius = 2
	dot_color = 'black'
	line_width = 1
	line_color = 'red'
	###################################绘制第一个点的图像 #########################
	edge_idx = target_layer_idx[0].view(196, -1).cpu().numpy()     #第n个block的边矩阵
	selected_row_max = edge_idx[max_index]                       #选点的第一个点
	selected_data_max = []
	# 遍历选定行中的值
	for value in selected_row_max:
		# 根据值选择相应的第二组数据项，并添加到 selected_data 中
		for item2 in small_images_with_index:
			if item2[1] == value:
				selected_data_max.append(item2)
	selected_data_max = np.array(selected_data_max, dtype=object)
	line_image = visual_image
	draw = ImageDraw.Draw(line_image)

	rowindex = 0
	row_image, row_index = selected_data_max[rowindex]
	hang = row_index // 14
	lie = row_index % 14
	row_center_x = row_image.width // 2 + lie*17
	row_center_y = row_image.height // 2 + hang*17
	# 连接第一个点的中心和剩余其他点的中心
	for item in selected_data_max[1:9]:
		_, index = item
		image, idx = small_images_with_index[index]
		idx_hang = idx // 14
		idx_lie = idx % 14
		center_x = image.width // 2 + idx_lie * 17
		center_y = image.height // 2 + idx_hang * 17
		draw.line([(row_center_x, row_center_y), (center_x, center_y)],
				  fill=line_color, width=line_width)
		draw.ellipse([(center_x - dot_radius, center_y - dot_radius),
					  (center_x + dot_radius, center_y + dot_radius)],
					 fill=line_color)
	#绘制中心点
	square_size = 2 * dot_radius
	draw.rectangle([(row_center_x - square_size, row_center_y - square_size),
					(row_center_x + square_size, row_center_y + square_size)],
				   fill=line_color)
	#############################################################################################################
	selected_row_min = edge_idx[min_index]  # 第几个点
	selected_data_min = []
	# 遍历选定行中的值
	for value in selected_row_min:
		# 根据值选择相应的第二组数据项，并添加到 selected_data 中
		for item2 in small_images_with_index:
			if item2[1] == value:
				selected_data_min.append(item2)
	selected_data_min = np.array(selected_data_min, dtype=object)
	line_image = visual_image
	draw = ImageDraw.Draw(line_image)

	rowindex = 0  # center
	row_image, row_index = selected_data_min[rowindex]
	hang = row_index // 14
	lie = row_index % 14
	row_center_x = row_image.width // 2 + lie * 17
	row_center_y = row_image.height // 2 + hang * 17
	# 连接第一个点的中心和剩余其他点的中心
	for item in selected_data_min[1:9]:
		_, index = item
		image, idx = small_images_with_index[index]
		idx_hang = idx // 14
		idx_lie = idx % 14
		center_x = image.width // 2 + idx_lie * 17
		center_y = image.height // 2 + idx_hang * 17
		draw.line([(row_center_x, row_center_y), (center_x, center_y)], fill=dot_color, width=line_width)
		draw.ellipse([(center_x - dot_radius, center_y - dot_radius),
					  (center_x + dot_radius, center_y + dot_radius)],
					 fill=dot_color)
	# 中心点
	square_size = 2 * dot_radius
	draw.rectangle([(row_center_x - square_size, row_center_y - square_size),
					(row_center_x + square_size, row_center_y + square_size)],
				   fill=dot_color)
	# 显示可视化图像
	line_image_out = line_image
	plt.imshow(line_image_out)
	plt.axis('off')
	plt.title(f"block:{block_num + 1}", fontsize=16)
	plt.show()
	""""##############################################
	num_digits = 196
	# 初始化一个数组来统计每个数字的次数
	digit_counts = np.zeros(num_digits, dtype=int)
	# 遍历 data_images，统计每个数字的出现次数
	for digit in edge_idx:
		digit_counts[digit] += 1
	#print(digit_counts)
	# 打印每个数字的出现次数
	#for digit, count in enumerate(digit_counts):
		#print(f"Digit {digit}: {count} times")
	heat_image = visual_image
	# 创建可视化图像的绘图对象
	draw = ImageDraw.Draw(heat_image)
	# 将counts中的值映射到热力图的颜色
	max_count = np.max(digit_counts)
	for i, count in enumerate(digit_counts):
		# 计算热力图中的矩形区域
		x1 = (i % 14) * 17  # 20是每个图像的宽度加间隔
		y1 = (i // 14) * 17  # 20是每个图像的高度加间隔
		x2 = x1 + 17  # 图像的宽度
		y2 = y1 + 17  # 图像的高度
		# 计算热力图颜色的强度
		intensity = int((count / max_count) * 255)
		# 选择颜色
		color = (intensity, 0, 0)  # 这里使用红色表示热力
		# 在热力图上绘制矩形
		draw.rectangle([x1, y1, x2, y2], fill=color)

	plt.imshow(heat_image)
	plt.axis('off')
	#plt.show()"""


	return small_images_with_index, feature_map, line_image_out


# 可视化小图片及其间隙
if __name__ == '__main__':
	small_images_with_index, visual_image, line_image = point_graph_out(image_path=r'E:\QYB\Repetition\transformer\vig_pytorch\me\12.jpg')
	visual_image.save("visual_image.jpg")
	plt.figure(3)
	plt.imshow(line_image)
	plt.axis('off')
	plt.show()









