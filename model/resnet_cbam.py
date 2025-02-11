import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock


# ---------------------- CBAM 模块定义 ----------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
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
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ---------------------- 修改后的 BasicBlock ----------------------
class BasicBlockWithCBAM(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv2.out_channels)  # 在残差块内部添加 CBAM

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)  # 在残差连接前应用 CBAM

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---------------------- 修改后的 ResNet18 ----------------------
class CBAMResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载原始 ResNet18 的预训练模型
        original_resnet = torchvision.models.resnet18(pretrained=pretrained)

        # 替换所有 BasicBlock 为 BasicBlockWithCBAM
        self._replace_basic_blocks(original_resnet)

        # 继承原始 ResNet18 的其他层
        self.conv1 = original_resnet.conv1
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4
        self.avgpool = original_resnet.avgpool

    def _replace_basic_blocks(self, original_resnet):
        # 递归替换所有 BasicBlock 为 BasicBlockWithCBAM
        for name, module in original_resnet.named_children():
            if isinstance(module, nn.Sequential):
                new_blocks = []
                for block in module:
                    if isinstance(block, BasicBlock):
                        # 创建新的 BasicBlockWithCBAM，继承原始参数
                        new_block = BasicBlockWithCBAM(
                            block.conv1.in_channels,
                            block.conv1.out_channels,
                            stride=block.conv1.stride[0],
                            downsample=block.downsample
                        )
                        # 复制原始权重（除了新增的 CBAM）
                        new_block.load_state_dict(block.state_dict(), strict=False)
                        new_blocks.append(new_block)
                    else:
                        new_blocks.append(block)
                setattr(original_resnet, name, nn.Sequential(*new_blocks))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x


# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    model = CBAMResNet18(pretrained=True)
    x = torch.randn(2, 3, 16, 16)
    output = model(x)

    # 打印修改后的模型结构
    print("\n模型结构摘要:")
    print(model)
    print("输出形状:", output.shape)
