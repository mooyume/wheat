import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, lengths):
        device = x.device  # 获取输入张量所在的设备

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype).to(device)  # 隐藏状态初始化

        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype).to(device)  # 细胞状态初始化

        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False).to(device)

        packed_output, _ = self.lstm(packed_input, (h_0, c_0))  # LSTM 前向传播

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output


if __name__ == '__main__':
    # 示例数据：假设我们有一个 (10, 20) 的 Tensor，其中 10 是 batch size，20 是填充后的序列长度
    data = torch.tensor([[1, 2, 3, float('nan'), float('nan')],
                         [4, 5, 6, 7, 8],
                         [9, 10, float('nan'), float('nan'), float('nan')],
                         [11, 12, 13, 14, float('nan')]], dtype=torch.float)

    # 创建长度掩码，计算每个序列的实际长度
    lengths = [torch.sum(~torch.isnan(seq)).item() for seq in data]

    # 替换 NaN 值为 0
    data[torch.isnan(data)] = 0
    input_size = 1
    hidden_size = 64
    num_layers = 2

    model = LSTMModel(input_size, hidden_size, num_layers)

    # 改变输入数据的形状以匹配 LSTM 输入
    data = data.unsqueeze(-1)  # 形状从 (10, 20) 变为 (10, 20, 1)

    # 获取模型输出
    output = model(data, lengths)
    print("输出形状：", output.shape)
