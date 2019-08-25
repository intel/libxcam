import math

import torch.nn.functional as F
from torch import nn

def swish(x):
    return x * F.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel, out_channels, stride):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))


class SRGAN(nn.Module):
    def __init__(self, upsample_factor, num_channel=1, base_filter=64):
        super(SRGAN, self).__init__()
        self.n_residual_blocks = 16
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(num_channel, base_filter, kernel_size=9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), ResidualBlock(in_channels=base_filter, out_channels=base_filter, kernel=3, stride=1))

        self.conv2 = nn.Conv2d(base_filter, base_filter, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filter)

        for i in range(self.upsample_factor // 2):
            self.add_module('upsample' + str(i + 1), UpsampleBlock(base_filter))

        self.conv3 = nn.Conv2d(base_filter, num_channel, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(self.upsample_factor // 2):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return self.conv3(x)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
