from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.self_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.wavepool_1 = WavePool(1024)
        self.wavepool_2 = WavePool(1024)
        # self.wavepool_3 = WavePool(1024)
        # self.instance_norm = nn.InstanceNorm1d(169)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

        self._reset_parameters()

        # self.alpha = nn.Parameter(torch.tensor(1.0))
        # self.beta = nn.Parameter(torch.tensor(1.0))
        # self.gamma = nn.Parameter(torch.tensor(1.0))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    ### HiLo Wavelet Self-Attention
    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # print("tgt11111111111: ", tgt.shape)
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        B, HW, C = tgt.shape
        # print("B, HW, C: ", B, HW, C)
        # tgt2 [100, 1, 256]
        tgt2_1 = self.self_attn_1(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt2_2 = self.self_attn_2(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt2_3 = self.self_attn_3(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]

        # 将形状从 [64, 169, 1024] 变成 [64, 1024, 169]
        tgt2_1 = tgt2_1.permute(0, 2, 1)
        tgt2_2 = tgt2_2.permute(0, 2, 1)
        tgt2_3 = tgt2_3.permute(0, 2, 1)
        # tgt2_2 = tgt2_2.permute(0, 2, 1)

        hw = int(np.sqrt(HW))
        # print("hw: ", hw)
        tgt2_1 = tgt2_1.reshape(B, C, hw, hw)
        tgt2_2 = tgt2_2.reshape(B, C, hw, hw)
        tgt2_3 = tgt2_3.reshape(B, C, hw, hw)
        # print("tgt2_1 Tensor Shape: ", tgt2_1.shape)

        if hw % 2 != 0:
            tgt2_1 = F.pad(tgt2_1, (0, 1, 0, 1), mode='replicate')
            tgt2_2 = F.pad(tgt2_2, (0, 1, 0, 1), mode='replicate')
            tgt2_3 = F.pad(tgt2_3, (0, 1, 0, 1), mode='replicate')
            # tgt2_2 = F.pad(tgt2_2, (0, 1, 0, 1), mode='replicate')
            # print("tgt2_1 Tensor Shape: ", tgt2_1.shape)
            hw = hw + 1
        else:
            hw = hw

        ll_1, lh_1, hl_1, hh_1 = self.wavepool_1(tgt2_1)
        ll_2, lh_2, hl_2, hh_2 = self.wavepool_2(tgt2_2)
        # ll_3, lh_3, hl_3, hh_3 = self.wavepool_3(tgt2_3)
        # print("ll_1 Tensor Shape: ", ll_1.shape)

        hw_reshape = int(hw * hw / 4)
        ll_1, lh_1, hl_1, hh_1 = ll_1.reshape(B, C, hw_reshape), lh_1.reshape(B, C, hw_reshape), hl_1.reshape(B, C,
                                                                                                              hw_reshape), hh_1.reshape(
            B, C, hw_reshape)
        ll_2, lh_2, hl_2, hh_2 = ll_2.reshape(B, C, hw_reshape), lh_2.reshape(B, C, hw_reshape), hl_2.reshape(B, C,
                                                                                                              hw_reshape), hh_2.reshape(
            B, C, hw_reshape)
        # ll_3, lh_3, hl_3, hh_3 = ll_3.reshape(B, C, hw_reshape), lh_3.reshape(B, C, hw_reshape), hl_3.reshape(B, C,
        #                                                                                                       hw_reshape), hh_3.reshape(
        #     B, C, hw_reshape)
        # print("ll_1 Tensor Shape: ", ll_1.shape)

        out_1 = torch.cat((lh_1, hl_1, (hh_1 + hh_2)/2, (lh_2 + hl_2)/2), -1)
        # out_1 = torch.cat((lh_1, hl_1, hh_1, (lh_2+hl_2+hh_2)/3), -1)
        # out_1 = torch.cat((lh_1, hl_1, hh_1, (lh_2 + hl_2 + hh_2) / 3), -1)
        # out_1 = torch.cat((lh_1, lh_2, (hl_1 + hl_2) / 2, (hh_1 + hh_2) / 2), -1)        
        out_1 = out_1.reshape(B, C, hw, hw)[:, :, :13, :13]
        out_1 = out_1.reshape(B, C, 169)
        # print("out_1 Tensor Shape: ", out_1.shape)
        # out_2 = torch.cat((ll_2, ll_2, ll_1, ll_3), -1)
        out_2 = torch.cat((ll_2, ll_2, ll_1, (ll_1 + ll_2) / 2), -1)
        # out_2 = torch.cat((ll_2,ll_1,ll_2,(ll_1+ll_2)/2), -1)
        out_2 = out_2.reshape(B, C, hw, hw)[:, :, :13, :13]
        out_2 = out_2.reshape(B, C, 169)
        # out_3 = torch.cat((lh_3, hl_3, hh_3, (lh_2 + hl_2 + hh_2) / 3), -1)
        # out_3 = out_3.reshape(B, C, hw, hw)[:, :, :13, :13]
        # out_3 = out_3.reshape(B, C, 169)

        # out_1 = self.instance_norm(out_1)
        # out_3 = self.instance_norm(out_3)

        # tgt2 = (out_1 + out_2 + out_3) / 3
        tgt2 = (out_1 + out_2) / 2
        # tgt2 = (self.alpha * out_1 + self.beta * out_2 + self.gamma * out_3) / (
        #             self.alpha + self.beta + self.gamma + 1e-6)
        tgt2 = tgt2.permute(0, 2, 1)
        # print("tgt2 Tensor Shape: ", tgt2.shape)

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    ### HiLo Wavelet Self-Attention

    def forward(self, tgt):

        return self.forward_post(tgt)


def main():
    # 设置随机种子确保可复现性
    torch.manual_seed(42)

    # B, C, h, w = 64, 1024, 13, 13
    # input_tensor = torch.rand(B, C, h, w)
    B, N, C = 64, 169, 1024
    input_tensor = torch.rand(B, N, C)  # 随机生成一个输入张量
    print("input_tensor Shape: ", input_tensor.shape)

    # 创建一个 SelfAttentionLayer 实例，设定模型参数
    d_model = 1024  # 输入和输出的特征维度
    nhead = 8  # 多头注意力中的头数
    dropout = 0.1  # dropout 概率
    activation = "relu"  # 激活函数

    model = SelfAttentionLayer(d_model=d_model, nhead=nhead, dropout=dropout, activation=activation)

    output = model(input_tensor)

    # 输出结果
    print("Output Tensor Shape: ", output.shape)


if __name__ == "__main__":
    main()

