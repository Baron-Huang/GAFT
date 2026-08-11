from loss_funcs.mmd import MMD_loss
from loss_funcs.adv import LambdaSheduler
import torch
import numpy as np
from loss_funcs.adv import *
def one_hot(org_x = None, pre_dim = 3):
	one_x = np.zeros((org_x.shape[0], pre_dim))
	for i in range(org_x.shape[0]):
		max_index = np.argmax(org_x[i])
		one_x[i][max_index] = 1
	return one_x


class LMMDLoss(MMD_loss, LambdaSheduler):
	def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
				 gamma=1.0, max_iter=1000, **kwargs):
		'''
		Local MMD
		'''
		super(LMMDLoss, self).__init__()
		super(MMD_loss, self).__init__()
		self.num_class = 3

	def forward(self, source, target, source_label, target_logits, gpu_device):
		if self.kernel_type == 'linear':
			raise NotImplementedError("Linear kernel is not supported yet.")

		elif self.kernel_type == 'rbf':
			#target_logits = F.softmax(target, dim=1)
			batch_size = source.size()[0]
			weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
			weight_ss = torch.from_numpy(weight_ss).to(gpu_device) # B, B
			weight_tt = torch.from_numpy(weight_tt).to(gpu_device)
			weight_st = torch.from_numpy(weight_st).to(gpu_device)

			kernels = self.guassian_kernel(source, target,
										   kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
										   fix_sigma=self.fix_sigma)
			loss = 0
			if torch.sum(torch.isnan(sum(kernels))):
				return loss
			SS = kernels[:batch_size, :batch_size]
			TT = kernels[batch_size:, batch_size:]
			ST = kernels[:batch_size, batch_size:]

			loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
			# Dynamic weighting
			#lamb = self.lamb()
			#self.step()
			#loss = loss #* lamb
			return loss

	def cal_weight(self, source_label, target_logits):
		batch_size = source_label.size()[0]
		source_logits = source_label.cpu().data.max(1)[1].numpy()
		source_label = source_label.cpu().data.numpy()
		source_label_onehot = one_hot(source_label,pre_dim=3)
		#source_label_onehot = np.eye(self.num_class)[source_label]  # one hot

		source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
		source_label_sum[source_label_sum == 0] = 100
		source_label_onehot = source_label_onehot / source_label_sum  # label ratio

		# Pseudo label
		target_label = target_logits.cpu().data.max(1)[1].numpy()

		target_logits = target_logits.cpu().data.numpy()
		target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
		target_logits_sum[target_logits_sum == 0] = 100
		target_logits = target_logits / target_logits_sum

		weight_ss = np.zeros((batch_size, batch_size))
		weight_tt = np.zeros((batch_size, batch_size))
		weight_st = np.zeros((batch_size, batch_size))

		set_s = set(source_logits)
		set_t = set(target_label)
		count = 0
		for i in range(self.num_class):  # (B, C)
			if i in set_s and i in set_t:
				s_tvec = source_label_onehot[:, i].reshape(batch_size, -1)  # (B, 1)
				t_tvec = target_logits[:, i].reshape(batch_size, -1)  # (B, 1)

				ss = np.dot(s_tvec, s_tvec.T)  # (B, B)
				weight_ss = weight_ss + ss
				tt = np.dot(t_tvec, t_tvec.T)
				weight_tt = weight_tt + tt
				st = np.dot(s_tvec, t_tvec.T)
				weight_st = weight_st + st
				count += 1

		length = count
		if length != 0:
			weight_ss = weight_ss / length
			weight_tt = weight_tt / length
			weight_st = weight_st / length
		else:
			weight_ss = np.array([0])
			weight_tt = np.array([0])
			weight_st = np.array([0])
		return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')