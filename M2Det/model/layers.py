import torch
import torch.nn as nn
import torch.nn.functional as F

from model import utils


class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_planes)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        return x


class FFMv1(nn.Module):
    def __init__(self):
        super(FFMv1, self).__init__()
        self.conv1 = Conv(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv(256, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, input1, input2):
        x = torch.cat(input1, utils.upsample_add(input2))

        return x


class FFMv2(nn.Module):
    def __init__(self):
        super(FFMv2, self).__init__()
        self.conv = Conv(256, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return x


class TUM(nn.Module):
    def __init__(self):
        super(TUM, self).__init__()

        encoder = []
        decoder = []
        smooth = []
        for i in range(6):
            encoder.append(Conv(256, 256, kernel_size=3, stride=2, padding=1))
            decoder.append(Conv(256, 256, kernel_size=3, stride=1, padding=0))
            smooth.append(Conv(256, 256, kernel_size=1, stride=1, padding=0))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.smooth = nn.Sequential(*smooth)

    def forward(self, x):
        return x


class SFAM(nn.Module):
    def __init__(self):
        super(SFAM, self).__init__()

    def forward(self, x):
        return x
