import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedCrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size, d_model, num_heads=8, dropout=0.1):
        super(EnhancedCrossAttentionFusion, self).__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model必须能被num_整除"

        # 线性投影层
        self.query_layer = nn.Linear(hidden_size, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)

        # 多头注意力融合层
        self.fc = nn.Linear(d_model, d_model)

        # 门控机制
        self.gate_layer = nn.Linear(2 * d_model, d_model)

        # 正则化组件
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, transformer_output, lstm_output):
        batch_size = transformer_output.size(0)

        # 生成Query/Key/Value
        query = self.query_layer(lstm_output)  # (B, d_model)
        key = self.key_layer(transformer_output)
        value = self.value_layer(transformer_output)

        # 多头拆分
        def reshape(x):
            return x.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = map(reshape, [query, key, value])

        # 注意力计算
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 注意力Dropout

        # 上下文聚合
        attn_output = torch.matmul(attn_weights, value)  # (B, nh, 1, hd)

        # 多头拼接
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1)
        attn_output = self.fc(attn_output)  # (B, d_model)

        # 残差连接与归一化
        attn_output = self.dropout(attn_output)
        attn_output = self.norm(attn_output + transformer_output)

        # 门控融合
        gate = torch.sigmoid(self.gate_layer(torch.cat([attn_output, transformer_output], dim=1)))
        output = gate * attn_output + (1 - gate) * transformer_output

        return output


# 测试代码
if __name__ == '__main__':
    batch_size = 32
    d_model = 512
    hidden_size = 64

    transformer_output = torch.randn(batch_size, d_model)
    lstm_output = torch.randn(batch_size, hidden_size)

    fusion_module = EnhancedCrossAttentionFusion(hidden_size, d_model)
    fused_output = fusion_module(transformer_output, lstm_output)
    print(fused_output.shape)  # 输出形状: (32, 512)