##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## updated by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

affine_par = True

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, '../utils'))
# sys.path.append('/inplace_abn')
# from bn import InPlaceABN, InPlaceABNSync

BatchNorm2d = functools.partial(nn.BatchNorm2d)


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), dropout=0.1):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            BatchNorm2d(out_features),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear',align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class PSPModuleV2(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), dropout=0.1):
        super(PSPModuleV2, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, padding=0),
            InPlaceABNSync(out_features),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear',align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class PSPModuleV3(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), dropout=0.1):
        super(PSPModuleV3, self).__init__()
        self.scale_num = len(sizes)
        self.contexts = [
            nn.Parameter(torch.randn(1, features, size, size).type(torch.FloatTensor).cuda(), requires_grad=True) \
            for size in sizes]
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, features, out_features, size):
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(conv, bn)

    def forward(self, feats):
        batch_size, c, h, w = feats.size()
        priors = [feats]
        for i in range(self.scale_num):
            priors += [F.upsample(input=self.stages[i](self.contexts[i].repeat(batch_size, 1, 1, 1)), \
                                  size=(h, w), mode='bilinear',align_corners=True)]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class GlobalContextModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1)):
        super(GlobalContextModule, self).__init__()
        self.stage = self._make_stage(features, out_features, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        # bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        global_context = F.upsample(input=self.stage(feats), size=(h, w), mode='bilinear',align_corners=True)
        return global_context


class ASPPModule(nn.Module):
    """
    Reference:
        Use the self-attention block to replace the global average pooling operation.
        Deeplabv3, combine the dilated convolution with the global average pooling.
    """

    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(out_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            InPlaceABNSync(out_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            InPlaceABNSync(out_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            InPlaceABNSync(out_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_features * 4, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features)
        )

    def _cat_each(self, feat1, feat2, feat3, feat4, feat5, feat6):
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat3[i], feat4[i], feat5[i], feat6[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        # feat1 = F.upsample(self.context_avg(x), size=(h, w), mode='bilinear')
        # feat2 = self.context_atten(x)
        feat3 = self.conv1(x)
        feat4 = self.conv2(x)
        feat5 = self.conv3(x)
        feat6 = self.conv4(x)

        if isinstance(x, Variable):
            out = torch.cat((feat3, feat4, feat5, feat6), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat3, feat4, feat5, feat6)
        else:
            raise RuntimeError('unknown input type')

        bottle = self.bottleneck(out)
        return bottle
