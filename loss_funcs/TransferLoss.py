# import torch
# import torch.nn as nn
# from .spd_mmd import SPDMMDLoss
# from .riemann_lem import LEM
# from .riemann_jeffrey import Jeffrey
# from .mmd import MMDLoss
# from .lmmd import LMMDLoss
# from .coral import CORAL
# from .adv import AdversarialLoss
# from .daan import DAANLoss
# from .bnm import BNM
# # from mmd import *
# # from coral import *
# # from adv import *
# # from lmmd import *
# # from daan import *
# # from bnm import *
#
# class TransferLoss(nn.Module):
#     def __init__(self, loss_type, **kwargs):
#         super(TransferLoss, self).__init__()
#         self.loss_type = loss_type
#         if loss_type == "mmd":
#             self.loss_func = MMDLoss(**kwargs)
#         elif loss_type == "spd_mmd":
#             self.loss_func = SPDMMDLoss(**kwargs)
#         elif loss_type == "lmmd":
#             self.loss_func = LMMDLoss(**kwargs)
#         elif loss_type == "coral":
#             self.loss_func = CORAL
#         elif loss_type == "adv":
#             self.loss_func = AdversarialLoss(**kwargs)
#         elif loss_type == "daan":
#             self.loss_func = DAANLoss(**kwargs)
#         elif loss_type == "bnm":
#             self.loss_func = BNM
#         elif loss_type == "lem":
#             self.loss_func = LEM(**kwargs)
#         elif loss_type == "jeffrey":
#             self.loss_func = Jeffrey(**kwargs)
#
#         else:
#             print("WARNING: No valid transfer loss function is used.")
#             self.loss_func = lambda x, y: 0 # return 0
#
#     def forward(self, source, target, **kwargs):
#         return self.loss_func(source, target, **kwargs)
import torch
import torch.nn as nn
import torch.nn.functional as F
from .spd_mmd import SPDMMDLoss
from .riemann_lem import LEM
from .riemann_jeffrey import Jeffrey
from .mmd import MMDLoss
from .lmmd import LMMDLoss
from .coral import CORAL
from .adv import AdversarialLoss
from .daan import DAANLoss
from .bnm import BNM

class TripletLoss(nn.Module):
    """
    三元组损失：拉近锚点与正例距离，拉远锚点与负例距离
    公式：L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin  # 边际参数，控制正负例距离差的阈值

    def forward(self, anchor, positive, negative):
        """
        参数:
            anchor: 锚点样本特征，形状 (batch_size, feature_dim)
            positive: 与锚点同类别的正例特征，形状 (batch_size, feature_dim)
            negative: 与锚点不同类别的负例特征，形状 (batch_size, feature_dim)
        返回:
            平均三元组损失
        """
        # 计算欧氏距离（也可替换为余弦距离等）
        dist_ap = F.pairwise_distance(anchor, positive, p=2)  # 锚点-正例距离
        dist_an = F.pairwise_distance(anchor, negative, p=2)  # 锚点-负例距离

        # 计算损失并取平均
        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        return loss


class TransferLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super(TransferLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "mmd":
            self.loss_func = MMDLoss(**kwargs)
        elif loss_type == "spd_mmd":
            self.loss_func = SPDMMDLoss(**kwargs)
        elif loss_type == "lmmd":
            self.loss_func = LMMDLoss(**kwargs)
        elif loss_type == "coral":
            self.loss_func = CORAL
        elif loss_type == "adv":
            self.loss_func = AdversarialLoss(**kwargs)
        elif loss_type == "daan":
            self.loss_func = DAANLoss(**kwargs)
        elif loss_type == "bnm":
            self.loss_func = BNM
        elif loss_type == "lem":
            self.loss_func = LEM(**kwargs)
        elif loss_type == "jeffrey":
            self.loss_func = Jeffrey(**kwargs)

        # 新增：三元组损失支持
        elif loss_type == "triplet":
            self.loss_func = TripletLoss(**kwargs)  # 传入margin等参数

        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda *args, **kwargs: 0  # 默认返回0

    def forward(self, *args, **kwargs):
        """
        适配不同损失的输入参数：
        - 普通损失（如mmd/coral）：输入(source, target, **kwargs)
        - 三元组损失：输入(anchor, positive, negative, **kwargs)
        """
        return self.loss_func(*args, **kwargs)