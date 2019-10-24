import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from base import BaseModel
from model import layers


class M2Det(BaseModel):

    def __init__(self, num_classes=10):
        super(M2Det, self).__init__()
        self.backbone = models.vgg16(pretrained=True).features()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, x):
        return x
