import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from base import BaseModel
from model import layers, utils


class M2Det(BaseModel):

    def __init__(self, num_classes=10, num_tums=8, phase="train"):
        super(M2Det, self).__init__()
        self.backbone = models.vgg16(pretrained=True).features()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.ffmv1 = layers.FFMv1()
        self.ffmv2 = layers.FFMv2()

        tum_layers = []
        for i in range(num_tums):
            tum_layers.append(layers.TUM())

        self.tum = nn.Sequential(*tum_layers)
        self.sfam = layers.SFAM()

        loc_ = list()
        conf_ = list()
        for i in range(self.num_scales):
            loc_.append(nn.Conv2d(self.planes * self.num_levels,
                                  4 * 6,  # 4 is coordinates, 6 is anchors for each pixels,
                                  3, 1, 1))
            conf_.append(nn.Conv2d(self.planes * self.num_levels,
                                   self.num_classes * 6,  # 6 is anchors for each pixels,
                                   3, 1, 1))
        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)

        self.phase = phase

    def forward(self, x):
        # Backbone network
        x = self.backbone(x)
        x = self.pool5(x)
        conv6 = self.conv6(x)
        conv6 = self.relu(conv6)
        conv7 = self.conv7(conv6)

        # MLFPN
        ## FFMv1
        base_feature = self.ffmv1(conv6, conv7)

        ## TUM
        input = base_feature
        output = []
        for idx, tum_layer in enumerate(self.tum):
            output.append(tum_layer(input))
            input = self.ffmv2(base_feature, output[idx])

        ## SFAM
        x = self.sfam(output)

        # Classification and Regression
        loc, conf = list(), list()
        for (x, l, c) in zip(x, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )

        return output
