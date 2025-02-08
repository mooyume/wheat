import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        # 将LSTM的输出投影到与Transformer相同的维度
        self.proj_lstm = nn.Linear(hidden_size, d_model)
        # 门控生成层：输入是拼接后的特征，输出门控向量
        self.gate_fc = nn.Linear(2 * d_model, d_model)

    def forward(self, transformer_output, lstm_output):
        # 投影LSTM输出到d_model维度
        projected_lstm = self.proj_lstm(lstm_output)
        # 拼接两个特征
        combined = torch.cat([transformer_output, projected_lstm], dim=1)
        # 生成门控向量（每个元素在0-1之间）
        gate = torch.sigmoid(self.gate_fc(combined))
        # 加权融合
        fused_output = gate * transformer_output + (1 - gate) * projected_lstm
        return fused_output


# 使用示例
batch_size = 32
d_model = 512
hidden_size = 64

# 初始化模块
fusion_module = GatedFusion(d_model, hidden_size)

# 生成随机输入
transformer_output = torch.randn(batch_size, d_model)
lstm_output = torch.randn(batch_size, hidden_size)

# 执行融合
fused_output = fusion_module(transformer_output, lstm_output)
print(fused_output.shape)  # 输出: torch.Size([32, 512])