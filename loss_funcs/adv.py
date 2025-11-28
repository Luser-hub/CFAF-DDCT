import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np


class LambdaSheduler(nn.Module):
    """
    用于动态调整对抗训练中梯度反转强度的调度器
    实现从0到1的平滑sigmoid增长，控制对抗损失的影响程度
    """

    def __init__(self, gamma=1.0, max_iter=1200, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma  # sigmoid函数的陡度参数，控制增长速度
        self.max_iter = max_iter  # 最大迭代次数，决定调度器何时达到最大值
        self.curr_iter = 0  # 当前迭代计数，用于跟踪训练进度

    def lamb(self):
        """计算当前迭代的lambda值（0到1之间）"""
        p = self.curr_iter / self.max_iter  # 迭代进度百分比
        # sigmoid函数变形：从0平滑增长到1
        # 初始阶段λ接近0，对抗训练影响小；训练后期λ接近1，对抗训练影响大
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        """更新迭代计数"""
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)


class AdversarialLoss(nn.Module):
    '''
    对抗性损失实现，基于梯度反转层(Gradient Reversal Layer)
    迫使模型学习对域差异不变的特征表示
    '''

    def __init__(self, input_dim=15750, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, **kwargs):
        super(AdversarialLoss, self).__init__()
        # 域判别器：区分特征是来自源域还是目标域
        self.domain_classifier = Discriminator(input_dim=input_dim, hidden_dim=256, reduction_dim=300)
        self.use_lambda_scheduler = use_lambda_scheduler  # 是否使用动态lambda调度
        if self.use_lambda_scheduler:
            # 初始化lambda调度器，控制梯度反转强度
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)

    def forward(self, source, target):
        """
        前向传播计算对抗损失
        source: 源域特征
        target: 目标域特征
        """
        lamb = 1.0  # 默认lambda值（若不使用调度器）
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()  # 获取当前迭代的lambda值
            self.lambda_scheduler.step()  # 更新迭代计数

        # 分别计算源域和目标域的对抗损失
        source_loss = self.get_adversarial_result(source, True, lamb)
        target_loss = self.get_adversarial_result(target, False, lamb)

        # 取平均作为最终对抗损失
        adv_loss = 0.5 * (source_loss + target_loss)
        return adv_loss

    def get_adversarial_result(self, x, source=True, lamb=1.0):
        """
        计算单个域的对抗损失
        x: 输入特征
        source: 是否为源域
        lamb: 梯度反转强度（核心参数）
        """
        x = ReverseLayerF.apply(x, lamb)  # 应用梯度反转层，关键步骤！
        domain_pred = self.domain_classifier(x)  # 通过判别器预测域标签

        # 根据域类型设置目标标签（源域为1，目标域为0）
        device = domain_pred.device
        if source:
            domain_label = torch.zeros(len(x), 1).long()
        else:
            domain_label = torch.ones(len(x), 1).long()

        # 计算二分类交叉熵损失（使用BCEWithLogitsLoss包含sigmoid）
        loss_fn = nn.BCEWithLogitsLoss()
        loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv


class ReverseLayerF(Function):
    """
    梯度反转层：在前向传播时不改变输入，在反向传播时反转梯度方向
    是对抗训练的核心机制，用于实现特征的域不变性
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha  # 保存梯度反转强度参数λ
        return x.view_as(x)  # 前向传播保持输入不变

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时反转梯度方向并乘以alpha(即λ)
        # λ控制梯度反转的强度：λ=0时无反转，λ=1时完全反转
        output = grad_output.neg() * ctx.alpha
        return output, None  # 返回反转后的梯度和None（对应alpha的梯度）


class Discriminator(nn.Module):
    """
    域判别器网络：区分输入特征是来自源域还是目标域
    """

    def __init__(self, input_dim=15750, hidden_dim=256, reduction_dim=300):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.reduction_dim = reduction_dim

        # 特征降维层（可选）
        self.reduction = nn.Sequential(
            nn.Linear(input_dim, reduction_dim),
            nn.BatchNorm1d(reduction_dim),
            nn.ReLU(),
        )

        # 判别器主体结构
        layers = [
            nn.Linear(reduction_dim, hidden_dim),  # 全连接层
            nn.BatchNorm1d(hidden_dim),  # 批量归一化
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 第二层全连接
            nn.BatchNorm1d(hidden_dim),  # 批量归一化
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_dim, 1),  # 输出层（二分类，预测域标签）
            # 移除Sigmoid激活，使用BCEWithLogitsLoss包含sigmoid
        ]
        self.layers = torch.nn.Sequential(*layers)  # 组装网络层

    def forward(self, x):
        """
        前向传播
        如果输入是多维张量，先展平为二维张量（batch_size, features）
        """
        if x.dim() > 2:
            x = x.contiguous().view(x.size(0), -1)  # 展平多维张量
        x = self.reduction(x)  # 特征降维
        return self.layers(x)  # 通过网络层计算输出