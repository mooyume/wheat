import math

import torch
import torchvision
from torch import nn

from attention.attention import Attention
from attention.time_step_att import TimeStepAttention
from config.config import option as opt
from model.kan import KAN
from model.resnet_se import SEResNet18


class Kansformer(nn.Module):
    def __init__(self, x_channels, y_channels):
        """
         replay mlp to kan
        """
        super(Kansformer, self).__init__()

        # 添加一个额外的卷积层来处理不同通道数的输入
        self.conv1_modis = nn.Conv2d(x_channels, 3, kernel_size=1)
        self.conv1_11a2 = nn.Conv2d(y_channels, 3, kernel_size=1)

        # CNN部分，使用预训练的ResNet
        if opt.use_se:
            self.cnn_09a1 = SEResNet18(pretrained=True)
            self.cnn_11a2 = SEResNet18(pretrained=True)
        else:
            resnet_09a1 = torchvision.models.resnet18(pretrained=True)
            modules = list(resnet_09a1.children())[:-1]  # 去掉最后的全连接层
            self.cnn_09a1 = nn.Sequential(*modules)

            resnet_11a2 = torchvision.models.resnet18(pretrained=True)
            tem_11a2 = list(resnet_11a2.children())[:-1]
            self.cnn_11a2 = nn.Sequential(*tem_11a2)

        d_model = 512
        # d_model = 256

        # Transformer部分
        self.pos_encoder_09a1 = PositionalEncoding(d_model)
        self.transformer_09a1 = CustomTransformerEncoder(d_model, opt.n_head, opt.n_layers)

        self.pos_encoder_11a1 = PositionalEncoding(d_model)
        self.transformer_11a2 = CustomTransformerEncoder(d_model, opt.n_head, opt.n_layers)
        self.time_step_att = TimeStepAttention(input_dim=d_model * 2, num_heads=opt.n_head)

        if opt.att:
            self.att = Attention(input_dim=d_model * 2)

        # 输出层
        layers = []
        sizes = [d_model * 2, d_model, d_model // 2, 1]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != len(sizes) - 2:  # 不在最后一层
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(0.3))
        if opt.kan:
            if opt.fc:
                sizes = [d_model * 2, d_model, d_model // 2]
                self.fc = nn.Linear(sizes[-1], 1)
            self.kan = KAN(sizes)
        else:
            self.mlp = nn.Sequential(*layers)

    def forward(self, x, y, fldas):
        batch_size, timesteps, c, h, w = x.size()
        x_in = x.view(batch_size * timesteps, c, h, w)
        x_in = self.conv1_modis(x_in)
        x_out = self.cnn_09a1(x_in)
        r_in_x = x_out.view(batch_size, timesteps, -1).permute(1, 0, 2)

        batch_size_y, timesteps_y, c_y, h_y, w_y = y.size()
        y_in = y.view(batch_size_y * timesteps_y, c_y, h_y, w_y)
        y_in = self.conv1_11a2(y_in)
        y_out = self.cnn_11a2(y_in)
        r_in_y = y_out.view(batch_size, timesteps, -1).permute(1, 0, 2)

        r_in_x = self.pos_encoder_09a1(r_in_x)
        r_in_y = self.pos_encoder_11a1(r_in_y)
        r_out_x = self.transformer_09a1(r_in_x)
        r_out_y = self.transformer_11a2(r_in_y)

        if opt.time_att:
            # 时间特征提取
            r_last = self.time_step_att(torch.cat((r_out_x, r_out_y), -1))
        else:
            r_last_x = r_out_x[-1, :, :]
            r_last_y = r_out_y[-1, :, :]
            r_last = torch.cat((r_last_x, r_last_y), -1)

        if opt.att:
            r_last = self.att(r_last)

        if opt.kan:
            output = self.kan(r_last)
            if opt.fc:
                output = self.fc(output)
        else:
            output = self.mlp(r_last)

        return output


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)

        # self.linear1 = nn.Linear(d_model, d_model * 4)
        # self.dropout = nn.Dropout(0.1)
        # self.linear2 = nn.Linear(d_model * 4, d_model)

        sizes = [d_model, d_model * 4, d_model]
        self.kan = KAN(sizes)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.kan(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        encoder_layer = CustomTransformerEncoderLayer(d_model, n_head)
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # 调整位置编码的形状
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 假设 x 的形状为 (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :, :]


if __name__ == '__main__':
    # d_model = 512
    nnT = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(512, nhead=opt.n_head), num_layers=opt.n_layers)
    #
    transformer_encoder = CustomTransformerEncoder(512, opt.n_head, opt.n_layers)
    #
    print(nnT)
    print(transformer_encoder)
    src = torch.rand(10, 32, 512)
    a = nnT(src)
    b = transformer_encoder(src)
    print(a.shape)
    print(b.shape)
    kan = Kansformer(3, 3)
    print(kan)
    # att = nn.MultiheadAttention(d_model, 8)
