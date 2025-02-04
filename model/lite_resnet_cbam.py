import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock


# ---------------------- 轻量化CBAM模块 ----------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=32):  # 增大压缩比
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):  # 减小卷积核
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], 1)))


class CBAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        return self.sa(x) * x


# ---------------------- 修改后的BasicBlock ----------------------
class BasicBlockWithCBAM(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv2.out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


# ---------------------- 轻量化ResNet18 ----------------------
class LiteCBAMResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        original_resnet = torchvision.models.resnet18(pretrained=pretrained)

        # 调整初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = nn.Identity()  # 移除最大池化

        # 减少每个layer的block数量
        self.layer1 = self._make_layer(original_resnet.layer1)
        self.layer2 = self._make_layer(original_resnet.layer2)
        self.layer3 = self._make_layer(original_resnet.layer3)
        self.layer4 = self._make_layer(original_resnet.layer4)

        # 加载预训练权重
        if pretrained:
            self._load_weights(original_resnet)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000) if pretrained else None

    def _make_layer(self, original_layer):
        """保留每个layer的第一个block"""
        blocks = []
        first_block = original_layer[0]
        new_block = BasicBlockWithCBAM(
            first_block.conv1.in_channels,
            first_block.conv1.out_channels,
            stride=first_block.conv1.stride[0],
            downsample=first_block.downsample
        )
        new_block.load_state_dict(first_block.state_dict(), strict=False)
        blocks.append(new_block)
        return nn.Sequential(*blocks)

    def _load_weights(self, original_resnet):
        """加载适配后的预训练权重"""
        # 加载后续层权重
        self.layer1[0].load_state_dict(original_resnet.layer1[0].state_dict(), strict=False)
        self.layer2[0].load_state_dict(original_resnet.layer2[0].state_dict(), strict=False)
        self.layer3[0].load_state_dict(original_resnet.layer3[0].state_dict(), strict=False)
        self.layer4[0].load_state_dict(original_resnet.layer4[0].state_dict(), strict=False)

        # 加载bn1权重
        self.bn1.load_state_dict(original_resnet.bn1.state_dict())

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    model = LiteCBAMResNet18(pretrained=True)
    x = torch.randn(2, 3, 16, 16)
    output = model(x)
    print("\n轻量化模型结构:")
    print(model)
    print("输出形状:", output.shape)
