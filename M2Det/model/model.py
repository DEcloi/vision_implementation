import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from base import BaseModel
from model import layers, utils


class M2Det(BaseModel):

    def __init__(self, num_classes=10, num_tums=8):
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

        return x
