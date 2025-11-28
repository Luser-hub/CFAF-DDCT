# Authors: Yonghao Song <eeyhsong@gmail.com>
# Modified by: yunzinan
#
# License: BSD (3-clause)
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor
import warnings

from Utils.base import EEGModuleMixin, deprecated_args

import torch
import torch.nn as nn
import warnings
from typing import Optional, Tuple, Union, List, Dict
from braindecode.models.base import EEGModuleMixin

import torch
import torch.nn as nn
import warnings
from typing import Optional, Tuple, Union, List, Dict
from braindecode.models.base import EEGModuleMixin

import torch
import torch.nn as nn
import warnings
from typing import Union, Dict
from braindecode.models.base import EEGModuleMixin


class _PatchEmbedding(nn.Module):
    """补丁嵌入模块，将EEG信号转换为嵌入向量"""

    def __init__(self, n_filters_time, filter_time_length, n_channels, pool_time_length, stride_avg_pool, drop_prob):
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters_time, (1, filter_time_length), padding=(0, filter_time_length // 2))
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, stride_avg_pool))
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(2)  # 移除通道维度
        x = x.transpose(1, 2)  # 转换为 (batch, time, features)
        return self.dropout(x)


class _ResidualAttention(nn.Module):
    """多头注意力Res模块，包含多头注意力和跳跃连接"""

    def __init__(self, emb_size, att_heads, att_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=att_heads,
            dropout=att_drop,
            batch_first=True  # 确保batch维度在前
        )
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Dropout(att_drop),
            nn.Linear(4 * emb_size, emb_size)
        )
        self.dropout = nn.Dropout(att_drop)

    def forward(self, x):
        # 第一个残差块：多头注意力
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)  # self-attention
        x = residual + self.dropout(attn_output)

        # 第二个残差块：前馈网络
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)

        return x


class _TransformerEncoder(nn.Module):
    """Transformer编码器，由多个ResidualAttention模块组成"""

    def __init__(self, att_depth, emb_size, att_heads, att_drop):
        super().__init__()
        self.layers = nn.ModuleList([
            _ResidualAttention(emb_size, att_heads, att_drop)
            for _ in range(att_depth)
        ])
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _FullyConnected(nn.Module):
    """全连接层，用于特征整合"""

    def __init__(self, final_fc_length):
        super().__init__()
        self.final_fc_length = final_fc_length
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)


class _FinalLayer(nn.Module):
    """最终分类层，使用直接属性定义而非Sequential"""

    def __init__(self, n_classes, return_features, add_log_softmax):
        super().__init__()
        self.return_features = return_features
        self.add_log_softmax = add_log_softmax
        # 直接定义线性层作为属性，不使用Sequential
        self.fc0 = nn.Linear(0, n_classes)  # 临时占位
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if self.return_features:
            return x
        x = self.fc0(x)  # 直接使用fc0属性
        if self.add_log_softmax:
            x = self.log_softmax(x)
        return x


class EEGConformer(EEGModuleMixin, nn.Module):
    """EEG Conformer.

    带多头注意力Res模块的卷积Transformer，用于EEG解码。
    """

    def __init__(
            self,
            n_outputs=None,
            n_chans=None,
            n_times=None,
            n_filters_time=40,
            filter_time_length=25,
            pool_time_length=75,
            pool_time_stride=15,
            drop_prob=0.5,
            att_depth=6,
            att_heads=10,
            att_drop_prob=0.5,
            final_fc_length='auto',
            return_features=False,
            chs_info=None,
            input_window_seconds=None,
            sfreq=None,
            add_log_softmax=True,
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            add_log_softmax=add_log_softmax,
        )
        # 更新映射字典以匹配新的层命名
        self.mapping = {
            'classification_head.fc.6.weight': 'final_layer.fc0.weight',
            'classification_head.fc.6.bias': 'final_layer.fc0.bias'
        }

        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        if not (self.n_chans <= 64):
            warnings.warn("This model has only been tested on no more " +
                          "than 64 channels. no guarantee to work with " +
                          "more channels.", UserWarning)

        self.patch_embedding = _PatchEmbedding(
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            n_channels=self.n_chans,
            pool_time_length=pool_time_length,
            stride_avg_pool=pool_time_stride,
            drop_prob=drop_prob)

        if final_fc_length == "auto":
            assert self.n_times is not None
            final_fc_length = self.get_fc_size()

        self.transformer = _TransformerEncoder(
            att_depth=att_depth,
            emb_size=n_filters_time,
            att_heads=att_heads,
            att_drop=att_drop_prob)

        self.fc = _FullyConnected(
            final_fc_length=final_fc_length)

        self.final_layer = _FinalLayer(
            n_classes=self.n_outputs,
            return_features=return_features,
            add_log_softmax=self.add_log_softmax
        )

        # 初始化最终层的全连接权重
        with torch.no_grad():
            # 确保在计算fc_size时模型处于评估模式
            self.eval()
            dummy_input = torch.ones(1, self.n_chans, self.n_times)
            dummy_output = self.forward(dummy_input, return_intermediate=True)
            fc_size = dummy_output['fc'].shape[1]
            # 重新初始化线性层以匹配正确的输入维度
            self.final_layer.fc0 = nn.Linear(fc_size, self.n_outputs)
            # 恢复训练模式
            self.train()

    def forward(self, x: torch.Tensor, return_intermediate=False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        x = torch.unsqueeze(x, dim=1)  # (batch, 1, chs, times)
        x = self.patch_embedding(x)  # (batch, times, emb_size)
        patch_features = x  # 保存Patch Embedding输出

        x = self.transformer(x)  # (batch, times, emb_size)
        transformer_features = x  # 保存Transformer输出

        x = self.fc(x)  # (batch, final_fc_length)
        fc_features = x  # 保存全连接层输出

        x = self.final_layer(x)  # 分类头输出

        if return_intermediate:
            return {
                'patch': patch_features,
                'transformer': transformer_features,
                'fc': fc_features,
                'logits': x
            }
        else:
            return x

    def get_fc_size(self):
        """计算全连接层的输入大小"""
        out = self.patch_embedding(torch.ones((1, 1, self.n_chans, self.n_times)))
        size_embedding_1 = out.cpu().data.numpy().shape[1]
        size_embedding_2 = out.cpu().data.numpy().shape[2]
        return size_embedding_1 * size_embedding_2


class _PatchEmbedding(nn.Module):
    """Patch Embedding.

    The authors used a convolution module to capture local features,
    instead of position embedding.

    Parameters
    ----------
    n_filters_time: int
        Number of temporal filters, defines also embedding size.
    filter_time_length: int
        Length of the temporal filter.
    n_channels: int
        Number of channels to be used as number of spatial filters.
    pool_time_length: int
        Length of temporal poling filter.
    stride_avg_pool: int
        Length of stride between temporal pooling filters.
    drop_prob: float
        Dropout rate of the convolutional layer.

    Returns
    -------
    x: torch.Tensor
        The output tensor of the patch embedding layer.
    """

    def __init__(
            self,
            n_filters_time,
            filter_time_length,
            n_channels,
            pool_time_length,
            stride_avg_pool,
            drop_prob,
    ):
        super().__init__()
        # input: (batch_size, 1, n_chans, n_times)
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, n_filters_time,
                      (1, filter_time_length), (1, 1)), # converge in the time dimension
            # (batch_size, n_filters_time, n_chans, *n_times)
            nn.Conv2d(n_filters_time, n_filters_time,
                      (n_channels, 1), (1, 1)), # converge in the space dimension
            # (batch_size, n_filters_time, 1, *n_times)
            nn.BatchNorm2d(num_features=n_filters_time),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, pool_time_length), # avgpool in the time dimension
                stride=(1, stride_avg_pool)
            ),
            # pooling acts as slicing to obtain 'patch' along the
            # time dimension as in ViT
            nn.Dropout(p=drop_prob),
        )

        # (batch_size, n_filters_time, 1, **n_times)
        self.projection = nn.Sequential(
            nn.Conv2d(
                n_filters_time, n_filters_time, (1, 1), stride=(1, 1)
            ),  # transpose, conv could enhance fiting ability slightly XXX: what???
            Rearrange("b d_model 1 seq -> b seq d_model"),
            # (batch_size, n_times, n_filters_time)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class _MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(
            self.queries(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        keys = rearrange(
            self.keys(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        values = rearrange(
            self.values(x), "b n (h d) -> b h n d", h=self.num_heads
        )
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class _ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class _FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class _TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, att_heads, att_drop, forward_expansion=4):
        super().__init__(
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _MultiHeadAttention(emb_size, att_heads, att_drop),
                    nn.Dropout(att_drop),
                )
            ),
            _ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    _FeedForwardBlock(
                        emb_size, expansion=forward_expansion,
                        drop_p=att_drop
                    ),
                    nn.Dropout(att_drop),
                )
            ),
        )


class _TransformerEncoder(nn.Sequential):
    """Transformer encoder module for the transformer encoder.

    Similar to the layers used in ViT.

    Parameters
    ----------
    att_depth : int
        Number of transformer encoder blocks.
    emb_size : int
        Embedding size of the transformer encoder.
    att_heads : int
        Number of attention heads.
    att_drop : float
        Dropout probability for the attention layers.

    """

    def __init__(self, att_depth, emb_size, att_heads, att_drop):
        super().__init__(
            *[
                _TransformerEncoderBlock(emb_size, att_heads, att_drop)
                for _ in range(att_depth)
            ]
        )


class _FullyConnected(nn.Module):
    def __init__(self, final_fc_length,
                 drop_prob_1=0.5, drop_prob_2=0.3, out_channels=256,
                 hidden_channels=32):
        """Fully-connected layer for the transformer encoder.

        Parameters
        ----------
        final_fc_length : int
            Length of the final fully connected layer.
        n_classes : int
            Number of classes for classification.
        drop_prob_1 : float
            Dropout probability for the first dropout layer.
        drop_prob_2 : float
            Dropout probability for the second dropout layer.
        out_channels : int
            Number of output channels for the first linear layer.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        add_log_softmax: bool
            Whether to add LogSoftmax non-linearity as the final layer.
        """

        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(final_fc_length, out_channels),
            nn.ELU(),
            nn.Dropout(drop_prob_1),
            nn.Linear(out_channels, hidden_channels),
            nn.ELU(),
            nn.Dropout(drop_prob_2),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class _FinalLayer(nn.Module):
    def __init__(self, n_classes, hidden_channels=32, return_features=False, add_log_softmax=True):
        """Classification head for the transformer encoder.

        Parameters
        ----------
        n_classes : int
            Number of classes for classification.
        hidden_channels : int
            Number of output channels for the second linear layer.
        return_features : bool
            Whether to return input features.
        add_log_softmax : bool
            Adding LogSoftmax or not.
        """

        super().__init__()
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_channels, n_classes),
        )
        self.return_features = return_features
        if add_log_softmax:
            classification = nn.LogSoftmax(dim=1)
        else:
            classification = nn.Identity()
        if not self.return_features:
            self.final_layer.add_module("classification", classification)

    def forward(self, x):
        if self.return_features:
            out = self.final_layer(x)
            return out, x
        else:
            out = self.final_layer(x)
            return out