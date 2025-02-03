import torch
import torch.nn as nn
import torchvision


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(SEResNet18, self).__init__()
        cnn_net = torchvision.models.resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(cnn_net.conv1, cnn_net.bn1, cnn_net.relu, cnn_net.maxpool, SEBlock(64))
        self.layer1 = self._make_layer(cnn_net.layer1, 64)
        self.layer2 = self._make_layer(cnn_net.layer2, 128)
        self.layer3 = self._make_layer(cnn_net.layer3, 256)
        self.layer4 = self._make_layer(cnn_net.layer4, 512)
        self.avgpool = cnn_net.avgpool

    def _make_layer(self, layer, channels):
        layers = []
        for block in layer:
            layers.append(block)
            layers.append(SEBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.layer0(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        return x


if __name__ == '__main__':
    input_tensor = torch.randn(32, 3, 224, 224)

    model = SEResNet18(pretrained=True)
    model = SEResNet18(pretrained=True)
    out = model(input_tensor)
    print(model)
    print('=======================================')
    resnet_09a1 = torchvision.models.resnet18(pretrained=True)
    modules = list(resnet_09a1.children())[:-1]  # 去掉最后的全连接层
    cnn_09a1 = nn.Sequential(*modules)
    out_2 = cnn_09a1(input_tensor)
    print(cnn_09a1)
    # 计算并打印模型的参数量（单位：百万）
    total_params = sum(param.numel() for param in model.parameters())
    total_params_in_millions = total_params / 1_000_000
    print(f"模型的总参数量: {total_params_in_millions:.2f}M")
