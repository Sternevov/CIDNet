import torch
import torch.nn as nn
from PIL import Image
from models.language_encoder import language_encoder
from models.visual_encoder import visual_encoder
from models.DViR.head import WeakREChead
from models.network_blocks import MultiScaleFusion
from torch.nn import ModuleList, Linear
import torch.nn.functional as F
from transformers import logging
from transformers import Dinov2Model
from models.DViR.wav_transformer import SelfAttentionLayer


logging.set_verbosity_error()


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.select_num = __C.SELECT_NUM
        self.visual_encoder = visual_encoder(__C).eval()
        self.lang_encoder = language_encoder(__C, pretrained_emb, token_size)

        self.linear_vs = nn.Linear(1024, __C.HIDDEN_SIZE)
        self.linear_ts = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_sub = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_attr = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_pos = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.head = WeakREChead(__C)
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), hiden_planes=1024, scaled=True)
        self.linear = nn.Linear(768, 1024)
            
        self.dino_model = Dinov2Model.from_pretrained('/root/autodl-tmp/DViN-main/dinov2-base')
        self.dino_model.eval()
        
        self.wavtransformer_s = SelfAttentionLayer(d_model=1024, nhead=8, dropout=0.1, activation="relu")
        # self.loss_gate = nn.Linear(__C.HIDDEN_SIZE, 3)  # 输入整体文本特征 -> 3个分支loss权重
        
        # self.linear_router = nn.Linear(1024, 4)
        self.linear_router = nn.Linear(1024, 2)
        self.lambda_sparse = 0.01
        self.linear_decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.projector_expert = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )
        self.projector_yolo = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128)
        )

        self.class_num = __C.CLASS_NUM
        
        if __C.VIS_FREEZE:
            self.frozen(self.visual_encoder)
        if __C.VIS_FREEZE:
            self.frozen(self.dino_model)

    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def re_normalize(self, x):

        # original normalization parameters
        original_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        original_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        # new normalization parameters
        new_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        new_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)

        # unnormalize
        x_unnormalized = x * original_std + original_mean

        # use new normalization parameters
        x_new_pro = (x_unnormalized - new_mean) / new_std

        return x_new_pro

    def forward(self, x, y):

        # Vision and Language Encoding
        with torch.no_grad():
            boxes_all, x_, boxes_sml = self.visual_encoder(x)
        y_ = self.lang_encoder(y)

        x_new_pro = self.re_normalize(x)

        resized_image_feature_dino = F.interpolate(x, size=(364, 364), mode='bilinear', align_corners=False)

        # load dino model
        with torch.no_grad():
            dino_feature = self.dino_model(resized_image_feature_dino).last_hidden_state.to(x.device)
        dino_feature = dino_feature[:, 1:, :]
        dino_feature = dino_feature.transpose(1, 2).contiguous().view(dino_feature.size(0), dino_feature.size(2), 26,
                                                                      26)
        dino_feature = F.avg_pool2d(dino_feature, kernel_size=2, stride=2)
        dino_feature = self.linear(dino_feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Vision Multi Scale Fusion
        s, m, l = x_
        x_input = [l, m, s]
        l_new, m_new, s_new = self.multi_scale_manner(x_input)

        s_new_original = s_new.clone()

        dynamic_features = [dino_feature, s_new_original]

        yolo_feature = F.adaptive_avg_pool2d(s_new, (1, 1)).permute(0, 2, 3, 1).squeeze(1)  # (64, 1,1024)

        # Calculate the probability distribution of router_logits
        router_logits = self.linear_router(yolo_feature.detach()).squeeze(1)
        router_logits = torch.softmax(router_logits, dim=-1)

        # top1_probs, top1_idx = torch.topk(router_logits, 2, dim=1)
        bs_top1_idx = s_new_original.size(0)
        top1_idx = torch.tensor([[0, 1]] * bs_top1_idx)

        selected_features = torch.stack(dynamic_features, dim=1)
        selected_feature1 = selected_features[torch.arange(selected_features.size(0)), top1_idx[:, 0]]

        # Get the value of router_logits for the corresponding index
        batch_size = router_logits.size(0)
        # print("router_logits.shape:", router_logits.shape)
        # print("top1_idx shape:", top1_idx.shape)
        # print("top1_idx values:", top1_idx)
        selected_logits0 = router_logits[torch.arange(batch_size), top1_idx[:, 0]]
        selected_logits1 = router_logits[torch.arange(batch_size), top1_idx[:, 1]]

        logits_sum = selected_logits0 + selected_logits1
        selected_logits0 = selected_logits0 / logits_sum
        selected_logits1 = selected_logits1 / logits_sum

        s_new = s_new * selected_logits0[:, None, None, None] + \
                selected_feature1 * selected_logits1[:, None, None, None]

        z_expert = self.projector_expert(selected_feature1)
        z_yolo = self.projector_yolo(s_new_original)

        temperature = 0.5
        batch_size = z_expert.size(0)
        z_expert_norm = F.normalize(z_expert, dim=1)
        z_yolo_norm = F.normalize(z_yolo, dim=1)
        similarity_matrix = torch.matmul(z_expert_norm, z_yolo_norm.T)
        positive_sample = torch.diag(similarity_matrix)
        labels = torch.arange(batch_size).to(x.device)
        criterion = nn.CrossEntropyLoss()
        logits = similarity_matrix / temperature
        loss_contrastive = criterion(logits, labels)
        x_ = [s_new, m_new, l_new]

        # Anchor Selection
        boxes_sml_new = []
        mean_i = torch.mean(boxes_sml[0], dim=2, keepdim=True)
        mean_i = mean_i.squeeze(2)[:, :, 4]
        vals, indices = mean_i.topk(k=int(self.select_num), dim=1, largest=True, sorted=True)
        bs, gridnum, anncornum, ch = boxes_sml[0].shape
        bs_, selnum = indices.shape
        box_sml_new = boxes_sml[0].masked_select(
            torch.zeros(bs, gridnum).to(boxes_sml[0].device).scatter(1, indices, 1).bool().unsqueeze(2).unsqueeze(
                3).expand(bs, gridnum, anncornum, ch)).contiguous().view(bs, selnum, anncornum, ch)
        boxes_sml_new.append(box_sml_new)

        batchsize, dim, h, w = x_[0].size()
        i_new = x_[0].view(batchsize, dim, h * w).permute(0, 2, 1)
        
        # 小波变换
        i_new = self.wavtransformer_s(i_new)
        wave_feature = i_new
        text_feature = y_['lang_feat']
        
        bs, gridnum, ch = i_new.shape
        i_new = i_new.masked_select(
            torch.zeros(bs, gridnum).to(i_new.device).scatter(1, indices, 1).
                bool().unsqueeze(2).expand(bs, gridnum, ch)).contiguous().view(bs, selnum, ch)
        
        recon_text_feature = self.linear_decoder(i_new)
        # pooling
        recon_text_feature_pooled = recon_text_feature.mean(dim=1, keepdim=True)
        recon_loss = F.mse_loss(recon_text_feature_pooled, y_['flat_lang_feat'].unsqueeze(1))

        # Anchor-based Contrastive Learning
        x_new = self.linear_vs(i_new)
        y_new = self.linear_ts(y_['flat_lang_feat'].unsqueeze(1))
        ysub_new = self.linear_sub(y_['flat_feat_subj'].unsqueeze(1))
        yattr_new = self.linear_attr(y_['flat_feat_attr'].unsqueeze(1))
        ypos_new = self.linear_pos(y_['flat_feat_pos'].unsqueeze(1))
        # # [B, H] -> [B, 3]
        # gate_logits = self.loss_gate(y_['flat_lang_feat'])  
        # # 用 softmax 保证权重和为 1
        # weights = torch.softmax(gate_logits, dim=-1)  # [B, 3]
        # # 按 batch 平均权重
        # w = weights.mean(0)  # [3]
        if self.training:
            # loss = w[0] *self.head(x_new, y_new)+w[1] *self.head(x_new, ysub_new)+w[2] *self.head(x_new, yattr_new)+self.head(x_new, ypos_new)
            loss = self.head(x_new, y_new, ysub_new, yattr_new, ypos_new)
            sparse_loss = torch.sum(router_logits * torch.log(router_logits + 1e-8), dim=1).mean()
            total_loss = loss + recon_loss + self.lambda_sparse * sparse_loss + loss_contrastive
            return total_loss
        else:
            predictions_s = self.head(x_new, y_new, ysub_new, yattr_new, ypos_new)
            predictions_list = [predictions_s]
            box_pred = get_boxes(boxes_sml_new, predictions_list, self.class_num)
            return box_pred
            # return box_pred, text_feature, y_['flat_subj'],y_['flat_attr'], y_['flat_pos']


def get_boxes(boxes_sml, predictionslist, class_num):
    batchsize = predictionslist[0].size()[0]
    pred = []
    for i in range(len(predictionslist)):
        mask = predictionslist[i].squeeze(1)
        masked_pred = boxes_sml[i][mask]
        refined_pred = masked_pred.view(batchsize, -1, class_num + 5)
        refined_pred[:, :, 0] = refined_pred[:, :, 0] - refined_pred[:, :, 2] / 2
        refined_pred[:, :, 1] = refined_pred[:, :, 1] - refined_pred[:, :, 3] / 2
        refined_pred[:, :, 2] = refined_pred[:, :, 0] + refined_pred[:, :, 2]
        refined_pred[:, :, 3] = refined_pred[:, :, 1] + refined_pred[:, :, 3]
        pred.append(refined_pred.data)
    boxes = torch.cat(pred, 1)
    score = boxes[:, :, 4]
    max_score, ind = torch.max(score, -1)
    ind_new = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, 5)
    box_new = torch.gather(boxes, 1, ind_new)
    return box_new