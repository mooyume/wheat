import torch
import torch.nn as nn
import torch.nn.functional as F


class DualModalCrossGatingFusion(nn.Module):
    def __init__(self, d_model, hidden_size, dropout=0.2):
        """
        Args:
            d_model (int): 遥感影像特征的维度（Transformer输出维度）。
            hidden_size (int): 历史产量序列特征的维度。
            dropout (float): Dropout率。
        """
        super(DualModalCrossGatingFusion, self).__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size

        # 历史产量序列特征的投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 时间门控（T-Gate）和空间门控（S-Gate）的参数
        self.W_q_t = nn.Linear(d_model, d_model)  # 时间门控的Query
        self.W_k_s = nn.Linear(d_model, d_model)  # 时间门控的Key
        self.W_q_s = nn.Linear(d_model, d_model)  # 空间门控的Query
        self.W_k_t = nn.Linear(d_model, d_model)  # 空间门控的Key

        # 残差平衡系数
        self.lambda_residual = nn.Parameter(torch.tensor(0.5))  # 可学习的残差权重

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, F_rs, F_hist):
        """
        Args:
            F_rs (torch.Tensor): 遥感影像特征，形状为 (B, T, d_model)。
            F_hist (torch.Tensor): 历史产量序列特征，形状为 (B, hidden_size)。
        Returns:
            torch.Tensor: 融合后的特征，形状为 (B, T, d_model)。
        """

        # 1. 历史产量序列特征投影
        F_hist_proj = self.projection(F_hist)  # (B, d_model)

        # 2. 时间门控（T-Gate）
        Q_t = self.W_q_t(F_hist_proj)  # (B, T, d_model)
        K_s = self.W_k_s(F_rs)  # (B, T, d_model)
        G_t = torch.sigmoid(torch.matmul(Q_t, K_s.transpose(-1, -2)) / (self.d_model ** 0.5))  # (B, T, T)
        G_t = torch.mean(G_t, dim=-1, keepdim=True)  # (B, T, 1)

        # 3. 空间门控（S-Gate）
        Q_s = self.W_q_s(F_rs)  # (B, T, d_model)
        K_t = self.W_k_t(F_hist_proj)  # (B, T, d_model)
        G_s = torch.sigmoid(torch.matmul(Q_s, K_t.transpose(-1, -2)) / (self.d_model ** 0.5))  # (B, T, T)
        G_s = torch.mean(G_s, dim=-1, keepdim=True)  # (B, T, 1)

        # 4. 残差门控融合
        F_rs_gated = G_t * F_rs  # 时间门控遥感特征
        F_hist_gated = G_s * F_hist_proj  # 空间门控历史特征
        F_fused = F_rs_gated + F_hist_gated + self.lambda_residual * F_rs + (1 - self.lambda_residual) * F_hist_proj

        # 5. Dropout
        F_fused = self.dropout(F_fused)

        return F_fused


# 测试代码
if __name__ == "__main__":
    B, d_model, hidden_size = 32, 512, 64
    F_rs = torch.randn(B, d_model)  # 遥感影像特征
    F_hist = torch.randn(B, hidden_size)  # 历史产量序列特征

    model = DualModalCrossGatingFusion(d_model, hidden_size)
    output = model(F_rs, F_hist)
    print(output.shape)  # 输出形状应为 (B, T, d_model)
