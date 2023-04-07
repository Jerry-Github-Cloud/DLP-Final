import torch
import torch.nn as nn
import numpy as np


class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=9,
                stride=1,
                padding=9 // 2),
            nn.BatchNorm2d(
                num_features=32),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=3 // 2),
            nn.BatchNorm2d(
                num_features=64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=3 // 2),
            nn.BatchNorm2d(
                num_features=128),
            nn.ReLU())

        self.residual_block = ResidualBlock(n_channels=128, k_size=3, s=1)

        self.convT_1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            nn.BatchNorm2d(
                num_features=64),
            nn.ReLU())
        self.convT_2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            nn.BatchNorm2d(
                num_features=32),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=9,
                stride=1,
                padding=9 // 2),
            nn.BatchNorm2d(
                num_features=3),
            nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)
        x = self.residual_block(x)

        x = self.convT_1(x)
        x = self.convT_2(x)
        x = self.conv4(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, k_size, s):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels,
            kernel_size=k_size, stride=s, padding=k_size // 2)
        self.batch_norm1 = nn.BatchNorm2d(num_features=n_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=n_channels, out_channels=n_channels,
            kernel_size=k_size, stride=s, padding=k_size // 2)
        self.batch_norm2 = nn.BatchNorm2d(num_features=n_channels)

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        return x + y
