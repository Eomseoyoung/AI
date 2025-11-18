
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        


    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, 2, 1)
        self.left = nn.Sequential(
            ConvBNReLU(out_channels, out_channels // 2, 1, 1, 0),
            ConvBNReLU(out_channels // 2, out_channels, 3, 2, 1),
        )
        self.right = nn.MaxPool2d(3, stride=2, padding=1)
        self.fuse = ConvBNReLU(out_channels * 2, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        left = self.left(x)
        right = self.right(x)
        x = torch.cat([left, right], dim=1)
        return self.fuse(x)


class GatherExpansion(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        inter_channels = in_channels
        self.conv1 = ConvBNReLU(in_channels, inter_channels, 3, 1, 1)
        self.dwconv2 = nn.Conv2d(inter_channels, inter_channels, 3, stride, 1, groups=inter_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv_1x1 = nn.Conv2d(inter_channels, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.dwconv2(out)
        out = self.bn2(out)
        out = self.conv_1x1(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        return self.relu(out)


class SemanticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = StemBlock(3, 16)
        self.stage3 = GatherExpansion(16, 32, 2)
        self.stage4 = GatherExpansion(32, 64, 2)
        self.stage5 = GatherExpansion(64, 128, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


class BGALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail_down = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.semantic_up = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
        )

    def forward(self, detail, semantic):
        detail = self.detail_down(detail)
        semantic = F.interpolate(self.semantic_up(semantic), size=detail.shape[2:], mode='bilinear', align_corners=False)
        out = detail + semantic
        return F.relu(out)


class BiSeNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.detail = nn.Sequential(
            ConvBNReLU(3, 64, 3, 2, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 96, 3, 2, 1),
            ConvBNReLU(96, 96, 3, 1, 1),
            ConvBNReLU(96, 96, 3, 1, 1),
            ConvBNReLU(96, 128, 3, 2, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 128, 3, 1, 1),
        )

        self.semantic = SemanticBranch()
        self.bga = BGALayer()

        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        detail_feat = self.detail(x)
        semantic_feat = self.semantic(x)
        fusion = self.bga(detail_feat, semantic_feat)
        out = self.head(fusion)
        out = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=False)
        return out


