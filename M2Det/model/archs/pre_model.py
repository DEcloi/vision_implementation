import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.detect import Detect
from model.priors import PriorBox
from base import BaseModel


def upsample_add(x, y, fuse_type='interp'):
    _, _, H, W = x.size()
    if fuse_type == 'interp':
        return F.interpolate(y, size=(H, W), mode='bilinear') + x
    else:
        raise NotImplementedError


class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class FFMv1(nn.Module):
    def __init__(self):
        super(FFMv1, self).__init__()
        self.reduce = Conv(512, 256, kernel_size=3, stride=1, padding=1)
        self.up_reduce = Conv(1024, 512, kernel_size=1, stride=1)

    def forward(self, x, y):
        return torch.cat((self.reduce(x), F.interpolate(self.up_reduce(y), scale_factor=2, mode='nearest')), 1)


class FFMv2(nn.Module):
    def __init__(self):
        super(FFMv2, self).__init__()
        self.conv = Conv(768, 128, kernel_size=1, stride=1)

    def forward(self, x, y):
        return torch.cat((self.conv(x), y), 1)


class TUM(nn.Module):
    def __init__(self, layer):
        super(TUM, self).__init__()

        encoder = []
        decoder = []
        smooth = []
        for i in range(5):
            if layer == 0 and i == 0:
                encoder.append(Conv(128, 256, kernel_size=3, stride=2, padding=1))
            else:
                if i == 4:
                    encoder.append(Conv(256, 256, kernel_size=3, stride=1, padding=0))
                else:
                    encoder.append(Conv(256, 256, kernel_size=3, stride=2, padding=1))

        for i in range(5):
            if layer == 0 and i == 4:
                decoder.append(Conv(256, 128, kernel_size=3, stride=1, padding=1))
            else:
                decoder.append(Conv(256, 256, kernel_size=3, stride=1, padding=1))

        for i in range(6):
            if layer == 0 and i == 5:
                smooth.append(Conv(128, 128, kernel_size=1, stride=1, padding=0))
            else:
                smooth.append(Conv(256, 128, kernel_size=1, stride=1, padding=0))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.smooth = nn.Sequential(*smooth)

    def forward(self, x):
        encoder_output = [x]
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            encoder_output.append(x)

        encoder_output.reverse()

        result = []
        for idx, j in enumerate(encoder_output):
            if idx == 0:
                decoder_output = self.decoder[idx](j)
                result.append(self.smooth[idx](j))
            elif idx == 5:
                y = upsample_add(j, decoder_output)
                result.append(self.smooth[idx](y))
            else:
                y = upsample_add(j, decoder_output)
                decoder_output = self.decoder[idx](y)
                result.append(self.smooth[idx](y))

        return result


class SFAM(nn.Module):
    def __init__(self, channel=128, num_level=8, num_scales=6, reduction=64):
        super(SFAM, self).__init__()
        # SEBlock
        self.channel = channel * num_level
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.ModuleList([nn.Conv2d(self.channel, self.channel // reduction, 1, 1, 0)] * num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.channel // reduction, self.channel, 1, 1, 0)] * num_scales)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = []
        for i in range(len(x[0])):
            outputs.append(torch.cat([x[j][i] for j in range(len(x))], 1))

        result = []
        for idx, output in enumerate(outputs):
            y = self.avg_pool(output)
            y = self.fc1[idx](y)
            y = self.relu(y)
            y = self.fc2[idx](y)
            y = self.sigmoid(y)
            result.append(output * y)

        return result


class M2Det(BaseModel):

    def __init__(self, num_classes=10, num_levels=8, num_scales=6, planes=256, phase="train"):
        super(M2Det, self).__init__()
        self.phase = phase
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.num_classes = num_classes

        base_model = models.vgg16(pretrained=True)
        self.backbone = base_model.features
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.backbone_list = [n for n in self.backbone]
        self.backbone_list.pop()
        self.backbone_list.append(self.pool5)
        self.backbone_list.append(self.conv6)
        self.backbone_list.append(self.relu)
        self.backbone_list.append(self.conv7)
        self.backbone_list.append(self.relu)
        self.layer = nn.ModuleList(self.backbone_list)

        self.ffmv1 = FFMv1()
        self.ffmv2 = FFMv2()

        self.top_conv = Conv(768, 128, kernel_size=1, stride=1)
        tum_layers = []
        for i in range(self.num_levels):
            if i == 0:
                tum_layers.append(TUM(layer=0))
            else:
                tum_layers.append(TUM(layer=1))

        self.tum = nn.Sequential(*tum_layers)
        self.sfam = SFAM()

        loc_ = list()
        conf_ = list()
        for i in range(self.num_scales):
            loc_.append(nn.Conv2d(1024, 32, 3, 1, 1))
            conf_.append(nn.Conv2d(1024, self.num_classes * 6, 3, 1, 1))

        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)

    def forward(self, x):
        size = x.size()[2:]
        print(size)

        # Backbone network
        base_feats = list()
        # print(len(self.layer))
        # print(x.shape)
        for i in range(len(self.layer)):
            x = self.layer[i](x)
            # print(i, x.shape)
            if i == 22 or i == 34:
                base_feats.append(x)

        # x = self.pool5(x)
        # conv6 = self.conv6(x)
        # conv6 = self.relu(conv6)
        # conv7 = self.conv7(conv6)
        # conv7 = self.relu(conv7)

        # MLFPN
        ## FFMv1
        base_feature = self.ffmv1(base_feats[0], base_feats[1])

        ## TUM
        input = base_feature
        output = []
        for idx, tum_layer in enumerate(self.tum):
            if idx == 0:
                input = self.top_conv(input)
            output.append(tum_layer(input))
            input = self.ffmv2(base_feature, output[idx][-1])

        ## SFAM
        x = self.sfam(output)

        # Classification and Regression
        loc, conf = list(), list()
        for (x, l, c) in zip(x, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # feat_maps = []
        # for i in range(len(loc)):
        #     feat = []
        #     feat += [loc[i].size(1), loc[i].size(2)]
        #     feat_maps += [feat]

        prior_box = PriorBox(size, loc)
        with torch.no_grad():
            priors = prior_box.forward().cuda()

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                priors.type(type(x.data))
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                priors
            )

        return output
