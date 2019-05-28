import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import os
import sys
import pdb
import numpy as np
from torch.autograd import Variable
import functools
from collections import OrderedDict
affine_par = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append('/inplace_abn')
from modules import InPlaceABN, InPlaceABNSync

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')



class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class ASPPInPlaceABNBlock(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(56, 112),
                 up_ratio=2, aspp_sec=(12, 24, 36), norm_act=BatchNorm2d):
        super(ASPPInPlaceABNBlock, self).__init__()

        self.in_norm = norm_act(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                                                    ("conv1_0", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(size=feat_res, mode='bilinear'))]))

        self.conv1x1 = nn.Sequential(OrderedDict([("conv1_1", nn.Conv2d(in_chs, out_chs, kernel_size=1,
                                                                        stride=1, padding=0, bias=False,
                                                                        groups=1, dilation=1))]))

        self.aspp_bra1 = nn.Sequential(OrderedDict([("conv2_1", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[0], bias=False,
                                                                          groups=1, dilation=aspp_sec[0]))]))

        self.aspp_bra2 = nn.Sequential(OrderedDict([("conv2_2", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[1], bias=False,
                                                                          groups=1, dilation=aspp_sec[1]))]))

        self.aspp_bra3 = nn.Sequential(OrderedDict([("conv2_3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[2], bias=False,
                                                                          groups=1, dilation=aspp_sec[2]))]))

        self.aspp_catdown = nn.Sequential(OrderedDict([("norm_act", norm_act(5*out_chs)),
                                                       ("conv_down", nn.Conv2d(5*out_chs, out_chs, kernel_size=1,
                                                                               stride=1, padding=1, bias=False,
                                                                               groups=1, dilation=1)),
                                                       ("dropout", nn.Dropout2d(p=0.2, inplace=True))]))

        self.upsampling = nn.Upsample(size=(int(feat_res[0]*up_ratio), int(feat_res[1]*up_ratio)), mode='bilinear')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    # channel_shuffle: shuffle channels in groups
    # +++++++++++++++++++++++++++++++++++++++++++++++++++ #
    @staticmethod
    def _channel_shuffle(x, groups):
        """
        Channel shuffle operation
        :param x: input tensor
        :param groups: split channels into groups
        :return: channel shuffled tensor
        """
        batch_size, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batch_size, groups, channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous().view(batch_size, -1, height, width)

        return x

    def forward(self, x):
        x = self.in_norm(x)
        x = torch.cat([self.gave_pool(x),
                       self.conv1x1(x),
                       self.aspp_bra1(x),
                       self.aspp_bra2(x),
                       self.aspp_bra3(x)], dim=1)

        out = self.aspp_catdown(x)
        return out, self.upsampling(out)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False),
        BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
        BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilate, expand_ratio):
        """
        InvertedResidual: Core block of the MobileNetV2
        :param inp:    (int) Number of the input channels
        :param oup:    (int) Number of the output channels
        :param stride: (int) Stride used in the Conv3x3
        :param dilate: (int) Dilation used in the Conv3x3
        :param expand_ratio: (int) Expand ratio of the Channel Width of the Block
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels=inp, out_channels=inp * expand_ratio,
                      kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(inplace=True),

            # dw
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio,
                      kernel_size=3, stride=stride, padding=dilate, dilation=dilate,
                      groups=inp * expand_ratio, bias=False),
            BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(inplace=True),

            # pw-linear
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup,
                      kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        )

    def forward(self, x):
        if self.use_res_connect:
            return torch.add(x, 1, self.conv(x))
        else:
            return self.conv(x)