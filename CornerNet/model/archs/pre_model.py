import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from torchvision.models.resnet import resnet101


class convolution(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride=1):
        pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)

        return relu


class Corner_pooling(nn.Module):
    def __init__(self, stance):
        self.stance = stance

    def foward(self, x):
        if self.stance == 'left':
            rows, cols = x.shape
            for i in range(rows):
                max_value = x[i][cols-1]
                for j in range(cols-2, -1, -1):
                    if max_value > x[i][j]:
                        x[i][j] = max_value
                    else:
                        max_value = x[i][j]
            return x
        elif self.stance == 'right':
            rows, cols = x.shape
            for i in range(rows):
                max_value = x[i][0]
                for j in range(1, cols):
                    if max_value > x[i][j]:
                        x[i][j] = max_value
                    else:
                        max_value = x[i][j]
        elif self.stance == 'top':
            rows, cols = x.shape
            for i in range(cols-1, -1, -1):
                max_value = x[rows-1][i]
                for j in range(rows-2, -1, -1):
                    if max_value > x[j][i]:
                        x[j][i] = max_value
                    else:
                        max_value = x[j][i]
            return x
        elif self.stance == 'bottom':
            rows, cols = x.shape
            for i in range(cols-1, -1, -1):
                max_value = x[0][i]
                for j in range(1, rows):
                    if max_value > x[j][i]:
                        x[j][i] = max_value
                    else:
                        max_value = x[j][i]
            return x


class Corner_pooling_module(nn.Module):
    def __init__(self, dim, stance1, stance2):
        self.p_conv1 = convolution(dim, 128, 3)
        self.p_conv2 = convolution(dim, 128, 3)

        self.corner_pool1 = Corner_pooling(stance1)
        self.corner_pool2 = Corner_pooling(stance2)

        self.conv1 = nn.Conv2d(128, dim, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

        self.p_conv3 = convolution(dim, dim, 3)

    def forward(self, x):
        p_conv1 = self.p_conv1(x)
        p_conv2 = self.p_conv2(x)

        corner_pooling1 = self.corner_pool1(p_conv1)
        corner_pooling2 = self.corner_pool2(p_conv2)

        conv1 = self.conv1(corner_pooling1 + corner_pooling2)
        bn1 = self.bn1(conv1)

        conv2 = self.conv2(x)
        bn2 = self.bn2(conv2)
        relu = self.relu(bn1 + bn2)

        p_conv3 = self.p_conv3(relu)

        return p_conv3


class Prediction_module(nn.Module):
    def __init__(self, dim, stance1, stance2):
        self.cpm = Corner_pooling_module(dim, stance1, stance2)

        self.heatmaps_conv1 = nn.Conv2d(128, 128, kernel_size=3, bias=False)
        self.embeddings_conv1 = nn.Conv2d(128, 128, kernel_size=3, bias=False)
        self.offsets_conv1 = nn.Conv2d(128, 128, kernel_size=3, bias=False)

        self.heatmaps_relu = nn.ReLU(inplace=True)
        self.embeddings_relu = nn.ReLU(inplace=True)
        self.offsets_relu = nn.ReLU(inplace=True)

        self.heatmaps_conv2 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.embeddings_conv2 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.offsets_conv2 = nn.Conv2d(128, 128, kernel_size=1, bias=False)

    def forward(self, x):
        cpm = self.cpm(x)

        heatmaps_conv1 = self.heatmaps_conv1(cpm)
        heatmaps_relu = self.heatmaps_relu(heatmaps_conv1)
        heatmaps_conv2 = self.heatmaps_conv2(heatmaps_relu)

        embeddings_conv1 = self.embeddings_conv1(cpm)
        embeddings_relu = self.embeddings_relu(embeddings_conv1)
        embeddings_conv2 = self.embeddings_conv2(embeddings_relu)

        offsets_conv1 = self.offsets_conv1(cpm)
        offsets_relu = self.offsets_relu(offsets_conv1)
        offsets_conv2 = self.offsets_conv2(offsets_relu)

        return heatmaps_conv2, embeddings_conv2, offsets_conv2


class CornerNet(BaseModel):
    def __init__(self, dim):
        super().__init__()
        base_model = resnet101(pretrained=True)
        self.backbone = base_model.features
        self.tl_pred = Prediction_module(dim, 'top', 'left')
        self.br_pred = Prediction_module(dim, 'bottom', 'right')

    def forward(self, x):
        backbone = self.backbone(x)
        tl_heatmaps, tl_embeddings, tl_offsets = self.tl_pred(backbone)
        br_heatmaps, br_embeddings, br_offsets = self.br_pred(backbone)

        return tl_heatmaps, tl_embeddings, tl_offsets, br_heatmaps, br_embeddings, br_offsets
