import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 计算注意力权重
        attn_weights = self.attention(x)
        # 对输入特征进行加权
        weighted_output = x * attn_weights
        return weighted_output


if __name__ == '__main__':
    transformer2_output = torch.randn(16, 1024)
    attention = Attention(input_dim=1024)
    out = attention(transformer2_output)
    print(out.shape)
