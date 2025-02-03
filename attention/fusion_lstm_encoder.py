import torch
import torch.nn as nn
import torch.nn.functional as F


class QKVFusion(nn.Module):
    def __init__(self, hidden_size, d_model):
        super(QKVFusion, self).__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.query_layer = nn.Linear(hidden_size, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, transformer_output, lstm_output):
        # (batch_size, d_model) 作为键和值
        key = self.key_layer(transformer_output)  # (batch_size, d_model)
        value = self.value_layer(transformer_output)  # (batch_size, d_model)

        # LSTM 输出作为查询
        query = self.query_layer(lstm_output)  # (batch_size, d_model)

        # 扩展维度以匹配矩阵乘法的要求
        key = key.unsqueeze(1)  # (batch_size, 1, d_model)
        value = value.unsqueeze(1)  # (batch_size, 1, d_model)
        query = query.unsqueeze(1)  # (batch_size, 1, d_model)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)  # (batch_size, 1, 1)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, 1)

        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, 1, d_model)
        attention_output = attention_output.squeeze(1)  # (batch_size, d_model)

        # 融合后的输出
        output = self.fc(attention_output)  # (batch_size, d_model)

        return output


if __name__ == '__main__':
    # 示例调用
    batch_size = 32
    d_model = 128
    hidden_size = 64

    transformer_output = torch.randn(batch_size, d_model)  # (batch_size, d_model)
    lstm_output = torch.randn(batch_size, hidden_size)  # (batch_size, hidden_size)

    qkv_fusion = QKVFusion(hidden_size, d_model)
    fused_output = qkv_fusion(transformer_output, lstm_output)
    print(fused_output.shape)  # 输出形状: (batch_size, d_model)
