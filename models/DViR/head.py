# coding=utf-8
import torch
import torch.nn as nn


def getContrast(vis_emb, lan_emb, lan_sub_emb, lan_attr_emb, lan_pos_emb):
    sim_full = torch.einsum('avd, bqd -> baqv',vis_emb, lan_emb)
    sim_subj = torch.einsum('avd, bqd -> baqv', vis_emb, lan_sub_emb)
    sim_attr = torch.einsum('avd, bqd -> baqv', vis_emb, lan_attr_emb)
    sim_pos  = torch.einsum('avd, bqd -> baqv', vis_emb, lan_pos_emb)
    
    # weights = torch.softmax(weights, dim=0)  # [4], 总和为1
    sim_map = sim_full + sim_subj + sim_attr + sim_pos  # [B, N, L_max]
    # sim_map = (weights[0]*sim_full +
    #                weights[1]*sim_subj +
    #                weights[2]*sim_attr +
    #                weights[3]*sim_pos)
    batchsize = sim_map.shape[0]
    max_sims,_ = sim_map.topk(k=2, dim=-1, largest=True, sorted=True)
    max_sims = max_sims.squeeze(2)

    # Negative Anchor Augmentation
    max_sim_0,max_sim_1 = max_sims[...,0],max_sims[...,1]
    max_sim_1 = max_sim_1.masked_select(~torch.eye(batchsize).bool().to(max_sim_1.device)).contiguous().view(batchsize,batchsize-1)
    new_logits = torch.cat([max_sim_0,max_sim_1],dim=1)

    target = torch.eye(batchsize).to(vis_emb.device)
    target_pred = torch.argmax(target, dim=1)
    loss = nn.CrossEntropyLoss(reduction="mean")(new_logits, target_pred)
    return loss

def getPrediction(vis_emb, lan_emb,lan_sub_emb, lan_attr_emb, lan_pos_emb):
    sim_full = torch.einsum('bkd, byd -> byk', vis_emb, lan_emb)
    sim_subj = torch.einsum('bkd, byd -> byk', vis_emb, lan_sub_emb)
    sim_attr = torch.einsum('bkd, byd -> byk', vis_emb, lan_attr_emb)
    sim_pos = torch.einsum('bkd, byd -> byk', vis_emb, lan_pos_emb)
    sim_map = sim_full + sim_subj + sim_attr + sim_pos
    # weights = torch.softmax(weights, dim=0)  # [4], 总和为1
    # # sim_map = sim_full + sim_subj + sim_attr + sim_pos  # [B, N, L_max]
    # sim_map = (weights[0]*sim_full +
    #                weights[1]*sim_subj +
    #                weights[2]*sim_attr +
    #                weights[3]*sim_pos)
    maxval, v = sim_map.max(dim=2, keepdim=True)
    predictions = torch.zeros_like(sim_map).to(sim_map.device).scatter(2,v.expand(sim_map.shape), 1).bool()
    return predictions

class WeakREChead(nn.Module):
    def __init__(self, __C):
        super(WeakREChead, self).__init__()
        # self.weights = nn.Parameter(torch.ones(4))  # 初始化为 1

    def forward(self, vis_fs,lan_fs, lan_sub_emb, lan_attr_emb,lan_pos_emb):
        if self.training:
            loss = getContrast(vis_fs, lan_fs,lan_sub_emb, lan_attr_emb, lan_pos_emb)
            return loss
        else:
            predictions = getPrediction(vis_fs, lan_fs,lan_sub_emb, lan_attr_emb, lan_pos_emb)
            return predictions










