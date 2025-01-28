import torch
import torch.nn as nn


class TimeStepAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(TimeStepAttention, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim

        # 使用 MultiheadAttention API
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=False)

    def forward(self, x):
        """
        x: Tensor of shape (seq_len, batch_size, input_dim)
        """
        # 使用 MultiheadAttention 计算注意力输出
        attn_output, attn_weights = self.attn(x, x,
                                              x)  # (seq_len, batch_size, input_dim), (batch_size, seq_len, seq_len)

        # 2. 计算加权输出
        # attn_output 的形状是 (seq_len, batch_size, input_dim)
        # attn_weights 的形状是 (batch_size, seq_len, seq_len)
        weighted_output = torch.bmm(attn_weights, attn_output.transpose(0, 1))  # (batch_size, seq_len, input_dim)

        # 3. 对时间维度 (seq_len) 进行池化（如求平均）
        pooled_output = weighted_output.mean(dim=1)  # (batch_size, input_dim)

        return pooled_output


if __name__ == '__main__':
    # 测试代码
    input_tensor = torch.randn(32, 16, 1024)  # (seq_len, batch_size, input_dim)

    # 创建时间步加权模块
    time_step_attention = TimeStepAttention(input_dim=1024, num_heads=8)

    # 执行前向计算
    output = time_step_attention(input_tensor)

    # 输出结果的形状
    print(output.shape)  # 应该是 (batch_size, input_dim) -> (16, 1024)
