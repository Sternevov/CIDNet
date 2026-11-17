import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAttentionFeatureSplit111(nn.Module):
    def __init__(self, dim, tau=2.0):
        super().__init__()
        self.tau = tau

        self.attn1 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )
        self.attn2 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )

        # 可选：更稳的初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, T, C]
        """

        # Temperature-controlled temporal attention
        A1 = F.softmax(self.attn1(x) / self.tau, dim=1)   # [B,T,1]
        A2 = F.softmax(self.attn2(x) / self.tau, dim=1)   # [B,T,1]

        out1 = x * A1
        out2 = x * A2

        return out1, out2, A1, A2


class DualAttentionFeatureSplit22222(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()

        self.attn1 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)   # ← 输出 C
        )

        self.attn2 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim)   # ← 输出 C
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, T, C]
        """

        A1 = self.attn1(x)      # [B, T, C]
        A2 = self.attn2(x)      # [B, T, C]

        # F.softmax(A1, dim=-1)对 C 维度做 softmax，让每个时间步的通道权重归一
        A1 = F.softmax(A1, dim=-1)
        A2 = F.softmax(A2, dim=-1)

        out1 = x * A1          # [B, T, C]
        out2 = x * A2          # [B, T, C]

        return out1, out2, A1, A2


class DualAttentionFeatureSplit(nn.Module):
    def __init__(self, dim, tau=2.0, alpha=0.3):
        super().__init__()
        self.tau = tau
        self.alpha = alpha

        self.attn1 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )
        self.attn2 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )

        self.channel = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        A1 = F.softmax(self.attn1(x) / self.tau, dim=1)
        A2 = F.softmax(self.attn2(x) / self.tau, dim=1)

        C = self.channel(x.mean(dim=1)).unsqueeze(1)
        C_tilde = self.alpha * C + (1 - self.alpha)

        out1 = x * A1 * C_tilde
        out2 = x * A2 * C_tilde

        return out1, out2, A1, A2



class DualAttentionFeatureSplit2(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # temporal attention
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )

        # channel gate
        self.channel = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # temporal
        A = F.softmax(self.attn(x), dim=1)   # [B,T,1]

        # channel
        C = self.channel(x.mean(dim=1))      # [B,C]
        C = C.unsqueeze(1)                   # [B,1,C]

        # out1 = x * A * C
        # out2 = x * (1 - A) * (1 - C)
        out1 = x * A
        out2 = x * (1 - A)

        return out1, out2


class DualAttentionFeatureSplit1(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()

        # Branch 1 Attention
        self.attn1 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )

        # Branch 2 Attention
        self.attn2 = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )

        # 初始化（更稳定）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, T, C]  e.g., [64,169,1024]
        """

        # Attention logits
        A1 = self.attn1(x)                  # [B,T,1]
        A2 = self.attn2(x)                  # [B,T,1]

        # Softmax attention weight
        A1 = F.softmax(A1, dim=1)
        A2 = F.softmax(A2, dim=1)

        # Weighted features, shape unchanged
        out1 = x * A1                       # [B,T,C]
        out2 = x * A2                       # [B,T,C]

        # Save attention for diversity loss
        return out1, out2, A1, A2


# ----------------------------
# 多样性损失（强制两个关注不同方面）
# ----------------------------
# def diversity_loss(A1, A2):
#     """
#     A1, A2: [B,T,1]
#     """
#     A1 = A1.squeeze(-1)   # [B,T]
#     A2 = A2.squeeze(-1)   # [B,T]

#     # 方法1：余弦相似度惩罚（越小越好，所以加上 abs）
#     cos_sim = F.cosine_similarity(A1, A2, dim=1).mean()
    

#     # 方法2：KL 散度（可选）
#     kl = F.kl_div(A1.log(), A2, reduction="batchmean")

#     return cos_sim + 0.1 * kl  # 可调权重



# def diversity_loss(A1, A2, eps=1e-6):
#     """
#     A1, A2: [B,T,1]，来自 softmax，可能含 0 → log(0) = -inf
#     本版本只做必要修复，保持第一种写法效果
#     """
#     A1 = A1.squeeze(-1)   # [B,T]
#     A2 = A2.squeeze(-1)   # [B,T]

#     # -------- 1. safe cosine similarity --------
#     # A1, A2 已是 softmax，不会全零，所以余弦稳定
#     cos_sim = F.cosine_similarity(A1, A2, dim=1).mean()

#     # -------- 2. safe KL divergence --------
#     # softmax 输出中 0 → 替换为 eps，防 log(0)
#     P = A1.clamp(min=eps)
#     Q = A2.clamp(min=eps)

#     kl = F.kl_div(P.log(), Q, reduction="batchmean")

#     return cos_sim + 0.1 * kl

# def diversity_loss(A1, A2, eps=1e-8):
#     """
#     A1, A2: [B, T, 1]
#     """
#     A1 = A1.squeeze(-1)  # [B,T]
#     A2 = A2.squeeze(-1)  # [B,T]

#     # 方法1：余弦相似度惩罚（越小越好，所以加上 abs）
#     cos_sim = F.cosine_similarity(A1, A2, dim=1).mean()

#     # 计算 A1.log() 时要检查是否有不合法值
#     valid_log_A1 = (A1 > 0).all()  # 检查 A1 中是否所有值都大于 0

#     if valid_log_A1:
#         # 方法2：KL 散度（可选）
#         kl = F.kl_div(A1.log(), A2, reduction="batchmean")
#     else:
#         # 如果A1中有非正数值，则不计算KL散度
#         kl = torch.tensor(0.0, device=A1.device)  # 设置为0

#     return cos_sim + 0.1 * kl  # 可调权重


def diversity_loss(A1, A2):
    """
    A1, A2: [B, T, 1]
    """
    A1 = A1.squeeze(-1)
    A2 = A2.squeeze(-1)

    cos_sim = F.cosine_similarity(A1, A2, dim=1).mean()
    
    # # 方法2：KL 散度（可选）
    # kl = F.kl_div(A1.log(), A2, reduction="batchmean")
    
    return cos_sim  # 越小越好
    # return cos_sim + 0.1 * kl
    


# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    B, T, C = 64, 169, 1024
    x = torch.randn(B, T, C)

    model = DualAttentionFeatureSplit(dim=C)
    out1, out2, A1, A2 = model(x)

    print("out1 shape:", out1.shape)
    print("out2 shape:", out2.shape)

    # diversity loss 用法示例
    loss_div = diversity_loss(A1, A2)
    print("div loss:", loss_div.item())
