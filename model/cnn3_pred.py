import torch
import torch.nn as nn
from einops import rearrange


class YieldPredictionModel(nn.Module):
    def __init__(self, time_steps=12, in_channels=3):
        super().__init__()
        self.time_steps = time_steps

        # 修改后的主干网络（保留时间维度）
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),  # 不在时间维度下采样

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),  # 保持时间维度不变

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),  # 保持时间维度不变

            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(512),
            nn.GELU(),
            nn.MaxPool3d((1, 2, 2)),  # 保持时间维度不变

            nn.Conv3d(512, 1024, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(1024),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((self.time_steps, 1, 1))  # 保持时间维度
        )

        # 时序Transformer
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1024,
                nhead=8,
                dim_feedforward=2048
            ),
            num_layers=2
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x, y):
        x = torch.cat((x, y), dim=2)
        # 输入形状: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> b c t h w')  # (B,C,T,H,W)

        # 提取时空特征（输出形状: (B,1024,T,1,1)）
        spatial_feat = self.backbone(x).squeeze(-1).squeeze(-1)  # (B,1024,T)

        # 时序建模（调整为Transformer需要的形状）
        temporal_feat = rearrange(spatial_feat, 'b d t -> t b d')  # (T,B,1024)
        temporal_feat = self.temporal_encoder(temporal_feat)
        temporal_feat = temporal_feat.mean(dim=0)  # (B,1024)

        # 预测
        output = self.regressor(temporal_feat)
        return output.squeeze()


# 测试输入输出
if __name__ == "__main__":
    # 输入形状: (batch_size=2, time_steps=12, channels=3, height=128, width=128)
    dummy_input = torch.randn(8, 32, 4, 128, 128)
    y = torch.randn(8, 32, 4, 128, 128)
    model = YieldPredictionModel(time_steps=12, in_channels=4 + 4)
    output = model(dummy_input, y)

    print("输入形状:", dummy_input.shape)  # torch.Size([2, 32, 4, 128, 128])
    print("输出形状:", output.shape)  # torch.Size([2])
