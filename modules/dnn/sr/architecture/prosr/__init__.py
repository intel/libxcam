from math import ceil, floor, log2
from collections import OrderedDict

import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProSR(nn.Module):
    """docstring for PyramidDenseNet"""

    def __init__(self, residual_denseblock, num_init_features, bn_size,
                 growth_rate, ps_woReLU, level_config, level_compression,
                 res_factor, max_num_feature, max_scale, **kwargs):
        super(ProSR, self).__init__()
        self.max_scale = max_scale
        self.n_pyramids = int(log2(self.max_scale))

        # used in curriculum learning, initially set to the last scale
        self.current_scale_idx = self.n_pyramids - 1

        self.residual_denseblock = residual_denseblock
        self.DenseBlock = _DenseBlock
        self.Upsampler = PixelShuffleUpsampler
        self.upsample_args = {'woReLU': ps_woReLU}

        denseblock_params = {
            'num_layers': None,
            'num_input_features': num_init_features,
            'bn_size': bn_size,
            'growth_rate': growth_rate,
        }

        num_features = denseblock_params['num_input_features']

        # Initiate network

        # each scale has its own init_conv
        for s in range(1, self.n_pyramids + 1):
            self.add_module('init_conv_%d' % s, Conv2d(3, num_init_features,
                                                       3))

        # Each denseblock forms a pyramid
        for i in range(self.n_pyramids):
            block_config = level_config[i]
            pyramid_residual = OrderedDict()

            # starting from the second pyramid, compress the input features
            if i != 0:
                out_planes = num_init_features if level_compression <= 0 else int(
                    level_compression * num_features)
                pyramid_residual['compression_%d' % i] = CompressionBlock(
                    in_planes=num_features, out_planes=out_planes)
                num_features = out_planes

            # serial connect blocks
            for b, num_layers in enumerate(block_config):
                denseblock_params['num_layers'] = num_layers
                denseblock_params['num_input_features'] = num_features
                # residual dense block used in ProSRL and ProSRGAN
                if self.residual_denseblock:
                    pyramid_residual['residual_denseblock_%d' %
                                     (b + 1)] = DenseResidualBlock(
                                         res_factor=res_factor,
                                         **denseblock_params)
                else:
                    block, num_features = self.create_denseblock(
                        denseblock_params,
                        with_compression=(b != len(block_config) - 1),
                        compression_rate=kwargs['block_compression'])
                    pyramid_residual['denseblock_%d' % (b + 1)] = block

            # conv before upsampling
            block, num_features = self.create_finalconv(
                num_features, max_num_feature)
            pyramid_residual['final_conv'] = block
            self.add_module('pyramid_residual_%d' % (i + 1),
                            nn.Sequential(pyramid_residual))

            # upsample the residual by 2 before reconstruction and next level
            self.add_module(
                'pyramid_residual_%d_residual_upsampler' % (i + 1),
                self.Upsampler(2, num_features, **self.upsample_args))

            # reconstruction convolutions
            reconst_branch = OrderedDict()
            out_channels = num_features
            reconst_branch['final_conv'] = Conv2d(out_channels, 3, 3)
            self.add_module('reconst_%d' % (i + 1),
                            nn.Sequential(reconst_branch))

        init_weights(self)

    def get_init_conv(self, idx):
        """choose which init_conv based on curr_scale_idx (1-based)"""
        return getattr(self, 'init_conv_%d' % idx)

    def forward(self, x, upscale_factor=None, blend=1.0):
        if upscale_factor is None:
            upscale_factor = self.max_scale
        else:
            valid_upscale_factors = [
                2**(i + 1) for i in range(self.n_pyramids)
            ]
            if upscale_factor not in valid_upscale_factors:
                error("Invalid upscaling factor {}: choose one of: {}".format(
                    upscale_factor, valid_upscale_factors))
                raise SystemExit(1)

        feats = self.get_init_conv(log2(upscale_factor))(x)
        for s in range(1, int(log2(upscale_factor)) + 1):
            if self.residual_denseblock:
                feats = getattr(self, 'pyramid_residual_%d' % s)(feats) + feats
            else:
                feats = getattr(self, 'pyramid_residual_%d' % s)(feats)
            feats = getattr(
                self, 'pyramid_residual_%d_residual_upsampler' % s)(feats)

            # reconst residual image if reached desired scale /
            # use intermediate as base_img / use blend and s is one step lower than desired scale
            if 2**s == upscale_factor or (blend != 1.0 and 2**
                                          (s + 1) == upscale_factor):
                tmp = getattr(self, 'reconst_%d' % s)(feats)
                # if using blend, upsample the second last feature via bilinear upsampling
                if (blend != 1.0 and s == self.current_scale_idx):
                    base_img = nn.functional.upsample(
                        tmp,
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=True)
                if 2**s == upscale_factor:
                    if (blend != 1.0) and s == self.current_scale_idx + 1:
                        tmp = tmp * blend + (1 - blend) * base_img
                    output = tmp

        return output

    def create_denseblock(self,
                          denseblock_params,
                          with_compression=True,
                          compression_rate=0.5):
        block = OrderedDict()
        block['dense'] = self.DenseBlock(**denseblock_params)
        num_features = denseblock_params['num_input_features']
        num_features += denseblock_params['num_layers'] * denseblock_params['growth_rate']

        if with_compression:
            out_planes = num_features if compression_rate <= 0 else int(
                compression_rate * num_features)
            block['comp'] = CompressionBlock(
                in_planes=num_features, out_planes=out_planes)
            num_features = out_planes

        return nn.Sequential(block), num_features

    def create_finalconv(self, in_channels, max_channels=None):
        block = OrderedDict()
        if in_channels > max_channels:
            block['final_comp'] = CompressionBlock(in_channels, max_channels)
            block['final_conv'] = Conv2d(max_channels, max_channels, (3, 3))
            out_channels = max_channels
        else:
            block['final_conv'] = Conv2d(in_channels, in_channels, (3, 3))
            out_channels = in_channels
        return nn.Sequential(block), out_channels

    def class_name(self):
        return 'ProSR'


class Conv2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__()
        p = 0
        conv_block = []
        kernel_size = args[2]
        dilation = kwargs.pop('dilation', 1)
        padding = kwargs.pop('padding', None)
        if padding is None:
            if isinstance(kernel_size, collections.Iterable):
                assert (len(kernel_size) == 2)
            else:
                kernel_size = [kernel_size] * 2

            padding = (floor((kernel_size[0] - 1) / 2),
                       ceil((kernel_size[0] - 1) / 2),
                       floor((kernel_size[1] - 1) / 2),
                       ceil((kernel_size[1] - 1) / 2))

        try:
            if kwargs['padding_type'] == 'REFLECTION':
                conv_block += [
                    nn.ReflectionPad2d(padding),
                ]
            elif kwargs['padding_type'] == 'ZERO':
                p = padding
            elif kwargs['padding_type'] == 'REPLICATE':
                conv_block += [
                    nn.ReplicationPad2d(padding),
                ]

        except KeyError as e:
            # use default padding 'REFLECT'
            conv_block += [
                nn.ReflectionPad2d(padding),
            ]
        except Exception as e:
            raise e

        conv_block += [
            nn.Conv2d(*args, padding=p, dilation=dilation, **kwargs)
        ]
        self.conv = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv(x)


class PixelShuffleUpsampler(nn.Sequential):
    """Upsample block with pixel shuffle"""

    def __init__(self, ratio, planes, woReLU=True):
        super(PixelShuffleUpsampler, self).__init__()
        assert ratio == 3 or log2(ratio) == int(log2(ratio))
        layers = []
        for i in range(int(log2(ratio))):
            if ratio == 3:
                mul = 9
            else:
                mul = 4
            layers += [Conv2d(planes, mul * planes, 3), nn.PixelShuffle(2)]
            if not woReLU:
                layers.append(nn.ReLU(inplace=True))

        self.m = nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """ResBlock"""

    #  ResBlock def __init__(self, blocks, planes, res_factor=1, act_type='RELU', act_params=dict()):
    def __init__(self,
                 block_type,
                 act_type,
                 planes,
                 res_factor=1,
                 act_params=dict()):
        super(ResidualBlock, self).__init__()
        self.block_type = block_type
        self.act_type = act_type
        self.res_factor = res_factor

        if self.block_type == block_type.BRCBRC:
            self.m = nn.Sequential(
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                Conv2d(planes, planes, 3),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                Conv2d(planes, planes, 3),
            )
        elif self.block_type == block_type.CRC:
            self.m = nn.Sequential(
                Conv2d(planes, planes, 3),
                nn.ReLU(inplace=True),
                Conv2d(planes, planes, 3),
            )
        elif self.block_type == block_type.CBRCB:
            self.m = nn.Sequential(
                Conv2d(planes, planes, 3),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                Conv2d(planes, planes, 3),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        return self.res_factor * self.m(x) + x


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        num_output_features = bn_size * growth_rate

        self.add_module(
            'conv_1',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=True)),

        self.add_module('relu_2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv_2',
            Conv2d(num_output_features, growth_rate, 3, stride=1, bias=True)),

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size)
            self.add_module('denselayer%d' % (i + 1), layer)


class DenseResidualBlock(nn.Sequential):
    def __init__(self, **kwargs):
        super(DenseResidualBlock, self).__init__()
        self.res_factor = kwargs.pop('res_factor')

        self.dense_block = _DenseBlock(**kwargs)
        num_features = kwargs['num_input_features'] + kwargs['num_layers'] * kwargs['growth_rate']

        self.comp = CompressionBlock(
            in_planes=num_features,
            out_planes=kwargs['num_input_features'],
        )

    def forward(self, x, identity_x=None):
        if identity_x is None:
            identity_x = x
        return self.res_factor * super(DenseResidualBlock,
                                       self).forward(x) + identity_x


class CompressionBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(CompressionBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = super(CompressionBlock, self).forward(x)
        if self.droprate > 0:
            out = F.dropout(
                out, p=self.droprate, inplace=False, training=self.training)
        return out


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        # m.weight.data.normal_(0, sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        b, c, h, w = m.weight.data.size()
        f = ceil(w / 2)
        cen = (2 * f - 1 - f % 2) / (2.0 * f)
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        fil = (1 - np.abs(xv / f - cen)) * (1 - np.abs(yv / f - cen))
        fil = fil[np.newaxis, np.newaxis, ...]
        fil = np.repeat(fil, 3, 0)
        m.weight.data.copy_(torch.from_numpy(fil))
