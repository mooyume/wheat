import torch
import torchvision
from torch import nn

from attention.attention import Attention
from attention.fusion_lstm_encoder import CrossAttentionFusion
from attention.time_step_att import TimeStepAttention
from config.config import option as opt
from model.custom_encoder import CustomTransformerEncoder, PositionalEncoding
from model.kan import KAN
from model.lstm import LSTMModel
from model.resnet_cbam import CBAMResNet18
from model.resnet_se import SEResNet18


class Kansformer_lstm(nn.Module):
    def __init__(self, x_channels, y_channels):
        """
         replay mlp to kan
         modis  9 channels
        """
        super(Kansformer_lstm, self).__init__()

        # 添加一个额外的卷积层来处理不同通道数的输入
        self.conv1_modis = nn.Conv2d(x_channels, 3, kernel_size=1)
        self.conv1_fldas = nn.Conv2d(y_channels, 3, kernel_size=1)

        # CNN部分，使用预训练的ResNet
        if opt.cnn_att == 'se':
            self.cnn_modis = SEResNet18(pretrained=True)
            self.cnn_fldas = SEResNet18(pretrained=True)
        elif opt.cnn_att == 'cbam':
            self.cnn_modis = CBAMResNet18(pretrained=True)
            self.cnn_fldas = CBAMResNet18(pretrained=True)
        else:
            resnet_09a1 = torchvision.models.resnet18(pretrained=True)
            modules = list(resnet_09a1.children())[:-1]  # 去掉最后的全连接层
            self.cnn_modis = nn.Sequential(*modules)

            resnet_11a2 = torchvision.models.resnet18(pretrained=True)
            tem_11a2 = list(resnet_11a2.children())[:-1]
            self.cnn_fldas = nn.Sequential(*tem_11a2)

        d_model = 512
        # d_model = 256

        self.time_step_att = TimeStepAttention(input_dim=d_model * 2, num_heads=opt.n_head)
        hidden_size = 64
        self.lstm = LSTMModel(1, hidden_size, 1)
        self.cross_fusion = CrossAttentionFusion(hidden_size, d_model * 2)
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

        if opt.struct == 'one_encoder':
            d_model = d_model * 2
            self.pos_encoder = PositionalEncoding(d_model)
            self.transformer = CustomTransformerEncoder(d_model, opt.n_head, opt.n_layers)
        if opt.struct == 'two_encoder':
            # Transformer部分
            self.pos_encoder_modis = PositionalEncoding(d_model)
            self.transformer_modis = CustomTransformerEncoder(d_model, opt.n_head, opt.n_layers)

            self.pos_encoder_fldas = PositionalEncoding(d_model)
            self.transformer_fldas = CustomTransformerEncoder(d_model, opt.n_head, opt.n_layers)

    def forward_lstm(self, h_data):
        # 创建长度掩码，计算每个序列的实际长度
        lengths = [torch.sum(~torch.isnan(seq)).item() for seq in h_data]

        # 替换 NaN 值为 0
        h_data[torch.isnan(h_data)] = 0

        h_data = h_data.unsqueeze(-1)
        output = self.lstm(h_data, lengths)
        return output[:, -1, :]

    def forward(self, x, y, fldas, h_data):
        if opt.struct == 'one_encoder':
            return self.forward_1encoder(x, y, fldas, h_data)
        if opt.struct == 'two_encoder':
            lstm_out = self.forward_lstm(h_data)
            modis_data = torch.cat([x, y], 2)
            batch_size, timesteps, c, h, w = modis_data.size()
            x_in = modis_data.view(batch_size * timesteps, c, h, w)
            x_in = self.conv1_modis(x_in)
            x_out = self.cnn_modis(x_in)
            r_in_x = x_out.view(batch_size, timesteps, -1).permute(1, 0, 2)

            batch_size_y, timesteps_y, c_y, h_y, w_y = fldas.size()
            y_in = fldas.view(batch_size_y * timesteps_y, c_y, h_y, w_y)
            y_in = self.conv1_fldas(y_in)
            y_out = self.cnn_fldas(y_in)
            r_in_y = y_out.view(batch_size, timesteps, -1).permute(1, 0, 2)

            r_in_x = self.pos_encoder_modis(r_in_x)
            r_in_y = self.pos_encoder_fldas(r_in_y)
            r_out_x = self.transformer_modis(r_in_x)
            r_out_y = self.transformer_fldas(r_in_y)

            if opt.time_att:
                # 时间特征提取
                r_last = self.time_step_att(torch.cat((r_out_x, r_out_y), -1))
            else:
                r_last_x = r_out_x[-1, :, :]
                r_last_y = r_out_y[-1, :, :]
                r_last = torch.cat((r_last_x, r_last_y), -1)

            # fusion lstm and encoder
            r_last = self.cross_fusion(r_last, lstm_out)

            if opt.att:
                r_last = self.att(r_last)

            if opt.kan:
                output = self.kan(r_last)
                if opt.fc:
                    output = self.fc(output)
            else:
                output = self.mlp(r_last)

            return output

    def forward_1encoder(self, x, y, fldas, h_data):
        lstm_out = self.forward_lstm(h_data)
        modis_data = torch.cat([x, y], 2)
        batch_size, timesteps, c, h, w = modis_data.size()
        x_in = modis_data.view(batch_size * timesteps, c, h, w)
        x_in = self.conv1_modis(x_in)
        x_out = self.cnn_modis(x_in)
        r_in_x = x_out.view(batch_size, timesteps, -1).permute(1, 0, 2)

        batch_size_y, timesteps_y, c_y, h_y, w_y = fldas.size()
        y_in = fldas.view(batch_size_y * timesteps_y, c_y, h_y, w_y)
        y_in = self.conv1_fldas(y_in)
        y_out = self.cnn_fldas(y_in)
        r_in_y = y_out.view(batch_size, timesteps, -1).permute(1, 0, 2)

        # fusion cnn
        r_in = torch.cat([r_in_x, r_in_y], 2)
        r_in = self.pos_encoder(r_in)
        r_out = self.transformer(r_in)

        if opt.time_att:
            # 时间特征提取
            r_last = self.time_step_att(r_out)
        else:
            r_last = r_out[-1, :, :]

        # fusion lstm and encoder
        r_last = self.cross_fusion(r_last, lstm_out)

        if opt.att:
            r_last = self.att(r_last)

        if opt.kan:
            output = self.kan(r_last)
            if opt.fc:
                output = self.fc(output)
        else:
            output = self.mlp(r_last)

        return output


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
    kan = Kansformer_lstm(3, 3)
    print(kan)
    # att = nn.MultiheadAttention(d_model, 8)
    a = torch.rand(10, 32, 512)
    b = torch.rand(10, 32, 512)
    c = torch.cat([a, b], 2)
    print(c.shape)
