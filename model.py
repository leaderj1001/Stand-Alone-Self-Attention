import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionConv, AttentionStem


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        self.stride = stride
        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            AttentionConv(width, width, kernel_size=7, padding=3, groups=8),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, self.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Model(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, stem=False):
        super(Model, self).__init__()
        self.in_places = 64

        if stem:
            self.init = nn.Sequential(
                # CIFAR10
                AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # For ImageNet
                # AttentionStem(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2, groups=1),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(4, 4)
            )
        else:
            self.init = nn.Sequential(
                # CIFAR10
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # For ImageNet
                # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                # nn.BatchNorm2d(64),
                # nn.ReLU(),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.dense = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_places, planes, stride))
            self.in_places = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out


def ResNet26(num_classes=1000, stem=False):
    return Model(Bottleneck, [1, 2, 4, 1], num_classes=num_classes, stem=stem)


def ResNet38(num_classes=1000, stem=False):
    return Model(Bottleneck, [2, 3, 5, 2], num_classes=num_classes, stem=stem)


def ResNet50(num_classes=1000, stem=False):
    return Model(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stem=stem)


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


# temp = torch.randn((2, 3, 224, 224))
# model = ResNet38(num_classes=1000, stem=True)
# print(get_model_parameters(model))
