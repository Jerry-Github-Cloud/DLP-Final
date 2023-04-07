import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class PerceptualLossNet(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.layers_name = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.layer1 = nn.Sequential()
        self.layer2 = nn.Sequential()
        self.layer3 = nn.Sequential()
        self.layer4 = nn.Sequential()

        for i in range(4):
            self.layer1.add_module(str(i), vgg16.features[i])
        for i in range(4, 9):
            self.layer2.add_module(str(i), vgg16.features[i])
        for i in range(9, 16):
            self.layer3.add_module(str(i), vgg16.features[i])
        for i in range(16, 23):
            self.layer4.add_module(str(i), vgg16.features[i])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.layer1(x)
        relu1_2 = x

        x = self.layer2(x)
        relu2_2 = x

        x = self.layer3(x)
        relu3_3 = x

        x = self.layer4(x)
        relu4_3 = x
        features = {
            'relu1_2': relu1_2,
            'relu2_2': relu2_2,
            'relu3_3': relu3_3,
            'relu4_3': relu4_3}
        return features
