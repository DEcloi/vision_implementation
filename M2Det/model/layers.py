import torch
import torch.nn as nn

from model import utils


class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
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

    def forward(self, x, y):
        return torch.cat(x, utils.upsample_add(y))


class FFMv2(nn.Module):
    def __init__(self):
        super(FFMv2, self).__init__()

    def forward(self, x,y ):
        return torch.cat(x, y)


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
        encoder_output = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            encoder_output.append(x)

        encoder_output.reverse()

        result = []
        for idx, j in enumerate(encoder_output):
            if idx == 0:
                decoder_output = self.decoder(j)
                result.append(self.smooth(j))
            else:
                y = utils.upsample_add(j, decoder_output)
                decoder_output = self.decoder[idx](y)
                result.append(self.smooth[idx](y))

        return result


class SFAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SFAM, self).__init__()
        # SEBlock
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = []
        for i in range(len(x[0])):
            for j in range(len(x)):
                output.append(torch.cat(x[j][i]))

        result = []
        for i in output:
            b, c, _, _ = i.size()
            y = self.avg_pool(i).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            result.append(y)

        return result
