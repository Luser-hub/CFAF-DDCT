from telnetlib import TSPEED
from .mmd import MMDLoss
from .adv import LambdaSheduler
import torch
import numpy as np

# 原版只考虑了源域数量与目标域数量相同的情况，下面进行了重写
# class LMMDLoss(MMDLoss, LambdaSheduler):
#     def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
#                     gamma=1.0, max_iter=1000, **kwargs):
#         '''
#         Local MMD
#         '''
#         super(LMMDLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
#         super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
#         self.num_class = num_class
#
#     def forward(self, source, target, source_label, target_logits):
#         if self.kernel_type == 'linear':
#             raise NotImplementedError("Linear kernel is not supported yet.")
#
#         elif self.kernel_type == 'rbf':
#             batch_size = source.size()[0]
#             weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
#             weight_ss = torch.from_numpy(weight_ss).cuda() # B, B
#             weight_tt = torch.from_numpy(weight_tt).cuda()
#             weight_st = torch.from_numpy(weight_st).cuda()
#
#             kernels = self.guassian_kernel(source, target,
#                                     kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#             loss = torch.Tensor([0]).cuda()
#             if torch.sum(torch.isnan(sum(kernels))):
#                 return loss
#             SS = kernels[:batch_size, :batch_size]
#             TT = kernels[batch_size:, batch_size:]
#             ST = kernels[:batch_size, batch_size:]
#
#             loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
#             # Dynamic weighting
#             lamb = self.lamb()
#             self.step()
#             loss = loss * lamb
#             return loss
#
#     def cal_weight(self, source_label, target_logits):
#         batch_size = source_label.size()[0]
#         source_label = source_label.cpu().data.numpy()
#         source_label_onehot = np.eye(self.num_class)[source_label] # one hot
#
#         source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
#         source_label_sum[source_label_sum == 0] = 100
#         source_label_onehot = source_label_onehot / source_label_sum # label ratio
#
#         # Pseudo label
#         target_label = target_logits.cpu().data.max(1)[1].numpy()
#
#         target_logits = target_logits.cpu().data.numpy()
#         target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
#         target_logits_sum[target_logits_sum == 0] = 100
#         target_logits = target_logits / target_logits_sum
#
#         weight_ss = np.zeros((batch_size, batch_size))
#         weight_tt = np.zeros((batch_size, batch_size))
#         weight_st = np.zeros((batch_size, batch_size))
#
#         set_s = set(source_label)
#         set_t = set(target_label)
#         count = 0
#         for i in range(self.num_class): # (B, C)
#             if i in set_s and i in set_t:
#                 s_tvec = source_label_onehot[:, i].reshape(batch_size, -1) # (B, 1)
#                 t_tvec = target_logits[:, i].reshape(batch_size, -1) # (B, 1)
#
#                 ss = np.dot(s_tvec, s_tvec.T) # (B, B)
#                 weight_ss = weight_ss + ss
#                 tt = np.dot(t_tvec, t_tvec.T)
#                 weight_tt = weight_tt + tt
#                 st = np.dot(s_tvec, t_tvec.T)
#                 weight_st = weight_st + st
#                 count += 1
#
#         length = count
#         if length != 0:
#             weight_ss = weight_ss / length
#             weight_tt = weight_tt / length
#             weight_st = weight_st / length
#         else:
#             weight_ss = np.array([0])
#             weight_tt = np.array([0])
#             weight_st = np.array([0])
#         return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


# class LMMDLoss(MMDLoss, LambdaSheduler):
#     def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
#                     gamma=1.0, max_iter=1000, **kwargs):
#         '''
#         Local MMD
#         '''
#         super(LMMDLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
#         super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
#         self.num_class = num_class
#
#     def forward(self, source, target, source_label, target_logits):
#         if self.kernel_type == 'linear':
#             raise NotImplementedError("Linear kernel is not supported yet.")
#
#         elif self.kernel_type == 'rbf':
#             s_batch_size = source.size()[0]
#             t_batch_size = target.size()[0]
#             weight_ss, weight_tt, weight_st, weight_ts = self.cal_weight(source_label, target_logits)
#             weight_ss = torch.from_numpy(weight_ss).cuda() # B, B
#             weight_tt = torch.from_numpy(weight_tt).cuda()
#             weight_st = torch.from_numpy(weight_st).cuda()
#             weight_ts = torch.from_numpy(weight_ts).cuda()
#
#             kernels = self.guassian_kernel(source, target,
#                                     kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#             loss = torch.Tensor([0]).cuda()
#             if torch.sum(torch.isnan(sum(kernels))):
#                 return loss
#             SS = kernels[:s_batch_size, :s_batch_size]
#             TT = kernels[s_batch_size:, s_batch_size:]
#             ST = kernels[:s_batch_size, s_batch_size:]
#             TS = kernels[s_batch_size:, :s_batch_size]
#
#             loss = loss + torch.sum( weight_ss * SS ) + torch.sum( weight_tt * TT ) - torch.sum( weight_st * ST ) - torch.sum( weight_ts * TS )
#             # Dynamic weighting
#             lamb = self.lamb()
#             self.step()
#             loss = loss * lamb
#             return loss
#
#     def cal_weight(self, source_label, target_logits):
#         s_batch_size = source_label.size()[0]
#         t_batch_size = target_logits.size()[0]
#         source_label = source_label.cpu().data.numpy()
#         source_label_onehot = np.eye(self.num_class)[source_label] # one hot
#
#         source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
#         source_label_sum[source_label_sum == 0] = 100
#         source_label_onehot = source_label_onehot / source_label_sum # label ratio
#
#         # Pseudo label
#         target_label = target_logits.cpu().data.max(1)[1].numpy()
#
#         target_logits = target_logits.cpu().data.numpy()
#         target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
#         target_logits_sum[target_logits_sum == 0] = 100
#         target_logits = target_logits / target_logits_sum
#
#         weight_ss = np.zeros((s_batch_size, s_batch_size))
#         weight_tt = np.zeros((t_batch_size, t_batch_size))
#         weight_st = np.zeros((s_batch_size, t_batch_size))
#         weight_ts = np.zeros((t_batch_size, s_batch_size))
#
#         set_s = set(source_label)
#         set_t = set(target_label)
#         count = 0
#         for i in range(self.num_class): # (B, C)
#             if i in set_s and i in set_t:
#                 s_tvec = source_label_onehot[:, i].reshape(s_batch_size, -1) # (B, 1)
#                 t_tvec = target_logits[:, i].reshape(t_batch_size, -1) # (B, 1)
#
#                 ss = np.dot(s_tvec, s_tvec.T) # (B, B)
#                 weight_ss = weight_ss + ss
#                 tt = np.dot(t_tvec, t_tvec.T)
#                 weight_tt = weight_tt + tt
#                 st = np.dot(s_tvec, t_tvec.T)
#                 weight_st = weight_st + st
#                 ts = np.dot(t_tvec, s_tvec.T)
#                 weight_ts = weight_ts + ts
#                 count += 1
#
#         length = count
#         if length != 0:
#             weight_ss = weight_ss / length
#             weight_tt = weight_tt / length
#             weight_st = weight_st / length
#             weight_ts = weight_ts / length
#         else:
#             weight_ss = np.array([0])
#             weight_tt = np.array([0])
#             weight_st = np.array([0])
#             weight_ts = np.array([0])
#         return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32'), weight_ts.astype('float32')

class LMMDLoss(MMDLoss, LambdaSheduler):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                 gamma=1.0, max_iter=1000):
        super(LMMDLoss, self).__init__(gamma, max_iter)
        self.num_class = num_class
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def forward(self, source, target, source_label, target_logits):
        batch_size = source.size(0)
        weight_ss, weight_tt, weight_st = self.cal_weight(
            source_label, target_logits,
            source_size=source.size(0),
            target_size=target.size(0)
        )

        # 转换为张量并移动至设备
        weight_ss = torch.from_numpy(weight_ss).to(source.device)
        weight_tt = torch.from_numpy(weight_tt).to(source.device)
        weight_st = torch.from_numpy(weight_st).to(source.device)

        # 计算核矩阵
        kernels = self.guassian_kernel(source, target, self.kernel_mul, self.kernel_num, self.fix_sigma)
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        # 计算加权MMD
        loss = torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        lamb = self.lamb()  # 获取动态权重
        self.step()  # 更新迭代次数
        return loss * lamb

    def cal_weight(self, source_label, target_logits, source_size, target_size):
        source_label = source_label.cpu().numpy()
        target_logits = target_logits.cpu().numpy()
        target_label = np.argmax(target_logits, axis=1)

        # 计算类别权重
        source_onehot = np.eye(self.num_class)[source_label]
        source_ratio = source_onehot / (np.sum(source_onehot, axis=0, keepdims=True) + 1e-8)

        target_ratio = target_logits / (np.sum(target_logits, axis=0, keepdims=True) + 1e-8)

        # 初始化权重矩阵
        weight_ss = np.zeros((source_size, source_size))
        weight_tt = np.zeros((target_size, target_size))
        weight_st = np.zeros((source_size, target_size))

        for cls in range(self.num_class):
            s_mask = (source_label == cls)
            t_mask = (target_label == cls)
            if np.sum(s_mask) == 0 or np.sum(t_mask) == 0:
                continue

            s_indices = np.where(s_mask)[0]  # 获取源域中属于当前类别的样本索引
            t_indices = np.where(t_mask)[0]  # 获取目标域中属于当前类别的样本索引

            s_weight = source_ratio[s_mask, cls].reshape(-1, 1)
            t_weight = target_ratio[t_mask, cls].reshape(-1, 1)

            # 使用获取的索引直接操作权重矩阵
            if len(s_indices) > 0:
                # 计算源域样本间的权重
                s_weight_matrix = np.dot(s_weight, s_weight.T)
                weight_ss[np.ix_(s_indices, s_indices)] += s_weight_matrix

            if len(t_indices) > 0:
                # 计算目标域样本间的权重
                t_weight_matrix = np.dot(t_weight, t_weight.T)
                weight_tt[np.ix_(t_indices, t_indices)] += t_weight_matrix

            # 计算源域和目标域样本间的权重
            if len(s_indices) > 0 and len(t_indices) > 0:
                st_weight_matrix = np.dot(s_weight, t_weight.T)
                weight_st[np.ix_(s_indices, t_indices)] += st_weight_matrix

        # 归一化权重
        valid_classes = np.sum(
            [(np.sum(source_label == cls) > 0) & (np.sum(target_label == cls) > 0) for cls in range(self.num_class)])
        weight_ss /= valid_classes + 1e-8
        weight_tt /= valid_classes + 1e-8
        weight_st /= valid_classes + 1e-8

        return weight_ss.astype(np.float32), weight_tt.astype(np.float32), weight_st.astype(np.float32)