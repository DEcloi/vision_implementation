import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from base import BaseModel
from model import layers


class M2Det(BaseModel):
    def __init__(self, num_classes=10):
        super(M2Det, self).__init__()

    def forward(self, x):
        return x
