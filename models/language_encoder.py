# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network_blocks import SA, AttFlat
from utils.utils import make_mask


class LSTM_SA(nn.Module):
    """
    多粒度文本编码器
    - 解耦主语(subj)、属性(attr)、位置(pos)
    - 每个粒度独立SA + AttFlat
    - 支持融合和多粒度输出，用于多粒度对比
    """
    def __init__(self, __C, pretrained_emb, token_size):
        super(LSTM_SA, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.GRU(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            dropout=__C.DROPOUT_R,
            bidirectional=False
        )

        # 三路 SA 分支
        self.sa_subj = SA(__C)
        self.sa_attr = SA(__C)
        self.sa_pos  = SA(__C)

        # 分支注意力门控 (Attention over SA)
        self.gate_proj = nn.Linear(__C.HIDDEN_SIZE, 3)  # 输出每个分支的权重

        # 每个粒度独立 AttFlat
        self.att_flat_subj  = AttFlat(__C)
        self.att_flat_attr  = AttFlat(__C)
        self.att_flat_pos   = AttFlat(__C)
        self.att_flat_merge = AttFlat(__C)

        if getattr(__C, "EMBED_FREEZE", False):
            self.frozen(self.embedding)

    def frozen(self, module):
        """冻结参数"""
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, ques_ix):
        """
        Args:
            ques_ix: [B,L] token idx
        Returns:
            dict with keys:
                flat_lang_feat: 融合后的文本特征
                lang_feat: LSTM 输出 + 加权 SA
                lang_feat_mask: mask
                flat_feat_subj/attr/pos: 各粒度特征
        """
        # 生成掩码
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))  # True=PAD
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)  # [B,L,H]

        # 三路 SA
        feat_subj = self.sa_subj(lang_feat, lang_feat_mask)
        feat_attr = self.sa_attr(lang_feat, lang_feat_mask)
        feat_pos  = self.sa_pos(lang_feat, lang_feat_mask)

        # 堆叠用于加权融合
        feats = torch.stack([feat_subj, feat_attr, feat_pos], dim=1)  # [B,3,L,H]

        # 用 LSTM 全局信息计算融合权重
        pooled = lang_feat.mean(1)  # [B,H]
        weights = F.softmax(self.gate_proj(pooled), dim=-1)  # [B,3]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B,3,1,1]

        # 加权融合
        lang_feat = (feats * weights).sum(1)  # [B,L,H]

        # 各粒度池化
        flat_feat_subj = self.att_flat_subj(feat_subj, lang_feat_mask)
        flat_feat_attr = self.att_flat_attr(feat_attr, lang_feat_mask)
        flat_feat_pos  = self.att_flat_pos(feat_pos, lang_feat_mask)
        flat_lang_feat = self.att_flat_merge(lang_feat, lang_feat_mask)

        return {
            'flat_lang_feat': flat_lang_feat,
            'lang_feat': lang_feat,
            'lang_feat_mask': lang_feat_mask,
            'flat_feat_subj': flat_feat_subj,
            'flat_feat_attr': flat_feat_attr,
            'flat_feat_pos': flat_feat_pos,
            'flat_subj': feat_subj,
            'flat_attr': feat_attr,
            'flat_pos': feat_pos,
        }


# registration
backbone_dict = {
    'lstm': LSTM_SA,
}


def language_encoder(__C, pretrained_emb, token_size):
    return backbone_dict[__C.LANG_ENC](__C, pretrained_emb, token_size)
