import torch
import torch.nn as nn
from .bn import ABN
from collections import OrderedDict


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        return inputs.view((in_size[0], in_size[1], 1, 1))


# +++++++++++++++++++++++++++++++++++++++++++++++++++ #
# InPlace Activated BatchNorm
# +++++++++++++++++++++++++++++++++++++++++++++++++++ #
class CatInPlaceABN(nn.Module):
    """
    Block for concat the two output tensor of feature net
    """
    def __init__(self, in_chs, norm_act=ABN):

        super(CatInPlaceABN, self).__init__()
        self.norm_act = norm_act(in_chs)

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        x = self.norm_act(x)
        return x


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# InplaceABNConv for Large Separable Convolution Block
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class LightHeadBlock(nn.Module):
    def __init__(self, in_chs, mid_chs=256, out_chs=256, kernel_size=15, norm_act=ABN):
        super(LightHeadBlock, self).__init__()
        pad = int((kernel_size - 1) / 2)

        # kernel size had better be odd number so as to avoid alignment error
        self.abn = norm_act(in_chs)
        self.conv_l = nn.Sequential(OrderedDict([("conv_lu", nn.Conv2d(in_chs, mid_chs,
                                                                       kernel_size=(kernel_size, 1),
                                                                       padding=(pad, 0))),
                                                 ("conv_ld", nn.Conv2d(mid_chs, out_chs,
                                                                       kernel_size=(1, kernel_size),
                                                                       padding=(0, pad)))]))

        self.conv_r = nn.Sequential(OrderedDict([("conv_ru", nn.Conv2d(in_chs, mid_chs,
                                                                       kernel_size=(1, kernel_size),
                                                                       padding=(0, pad))),
                                                 ("conv_rd", nn.Conv2d(mid_chs, out_chs,
                                                                       kernel_size=(kernel_size, 1),
                                                                       padding=(pad, 0)))]))

    def forward(self, x):
        x = self.abn(x)
        x_l = self.conv_l(x)
        x_r = self.conv_r(x)
        return torch.add(x_l, 1, x_r)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcs = nn.Sequential(nn.Linear(channel, int(channel/reduction)),
                                 nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                 nn.Linear(int(channel/reduction), channel),
                                 nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        y = self.avg_pool(x).view(bahs, chs)
        y = self.fcs(y).view(bahs, chs, 1, 1)
        return torch.mul(x, y)


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


class ModifiedSCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ModifiedSCSEBlock, self).__init__()
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

        spa_se = self.spatial_se(x)
        return torch.mul(torch.mul(x, chn_se), spa_se)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Vortex Pooling: Improving Context Representation in Semantic Segmentation
# https://arxiv.org/abs/1804.06242v1
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class VortexPooling(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2, rate=(3, 9, 27)):
        super(VortexPooling, self).__init__()
        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                                                    ("conv1x1", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(size=feat_res, mode='bilinear')),
                                                    ("bn0", nn.BatchNorm2d(num_features=out_chs))]))

        self.conv3x3 = nn.Sequential(OrderedDict([("conv3x3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                        stride=1, padding=1, bias=False,
                                                                        groups=1, dilation=1)),
                                                  ("bn3x3", nn.BatchNorm2d(num_features=out_chs))]))

        self.vortex_bra1 = nn.Sequential(OrderedDict([("avg_pool", nn.AvgPool2d(kernel_size=rate[0], stride=1,
                                                                                padding=int((rate[0]-1)/2), ceil_mode=False)),
                                                      ("conv3x3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                            stride=1, padding=rate[0], bias=False,
                                                                            groups=1, dilation=rate[0])),
                                                      ("bn3x3", nn.BatchNorm2d(num_features=out_chs))]))

        self.vortex_bra2 = nn.Sequential(OrderedDict([("avg_pool", nn.AvgPool2d(kernel_size=rate[1], stride=1,
                                                                                padding=int((rate[1]-1)/2), ceil_mode=False)),
                                                      ("conv3x3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                            stride=1, padding=rate[1], bias=False,
                                                                            groups=1, dilation=rate[1])),
                                                      ("bn3x3", nn.BatchNorm2d(num_features=out_chs))]))

        self.vortex_bra3 = nn.Sequential(OrderedDict([("avg_pool", nn.AvgPool2d(kernel_size=rate[2], stride=1,
                                                                                padding=int((rate[2]-1)/2), ceil_mode=False)),
                                                      ("conv3x3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                            stride=1, padding=rate[2], bias=False,
                                                                            groups=1, dilation=rate[2])),
                                                      ("bn3x3", nn.BatchNorm2d(num_features=out_chs))]))

        self.vortex_catdown = nn.Sequential(OrderedDict([("conv_down", nn.Conv2d(5 * out_chs, out_chs, kernel_size=1,
                                                                                 stride=1, padding=1, bias=False,
                                                                                 groups=1, dilation=1)),
                                                         ("bn_down", nn.BatchNorm2d(num_features=out_chs)),
                                                         ("dropout", nn.Dropout2d(p=0.2, inplace=True))]))

        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio), int(feat_res[1] * up_ratio)), mode='bilinear')

    def forward(self, x):
        out = torch.cat([self.gave_pool(x),
                         self.conv3x3(x),
                         self.vortex_bra1(x),
                         self.vortex_bra2(x),
                         self.vortex_bra3(x)], dim=1)

        out = self.vortex_catdown(out)
        return self.upsampling(out)


class ASPPBlock(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(56, 112), up_ratio=2, aspp_sec=(12, 24, 36)):
        super(ASPPBlock, self).__init__()

        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                                                    ("conv1_0", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(size=feat_res, mode='bilinear')),
                                                    ("bn0", nn.BatchNorm2d(num_features=out_chs))]))

        self.conv1x1 = nn.Sequential(OrderedDict([("conv1_1", nn.Conv2d(in_chs, out_chs, kernel_size=1,
                                                                        stride=1, padding=0, bias=False,
                                                                        groups=1, dilation=1)),
                                                  ("bn1_1", nn.BatchNorm2d(num_features=out_chs))]))

        self.aspp_bra1 = nn.Sequential(OrderedDict([("conv2_1", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[0], bias=False,
                                                                          groups=1, dilation=aspp_sec[0])),
                                                    ("bn2_1", nn.BatchNorm2d(num_features=out_chs))]))

        self.aspp_bra2 = nn.Sequential(OrderedDict([("conv2_2", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[1], bias=False,
                                                                          groups=1, dilation=aspp_sec[1])),
                                                    ("bn2_2", nn.BatchNorm2d(num_features=out_chs))]))

        self.aspp_bra3 = nn.Sequential(OrderedDict([("conv2_3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                          stride=1, padding=aspp_sec[2], bias=False,
                                                                          groups=1, dilation=aspp_sec[2])),
                                                    ("bn2_3", nn.BatchNorm2d(num_features=out_chs))]))

        self.aspp_catdown = nn.Sequential(OrderedDict([("conv_down", nn.Conv2d(5*out_chs, out_chs, kernel_size=1,
                                                                               stride=1, padding=1, bias=False,
                                                                               groups=1, dilation=1)),
                                                       ("bn_down", nn.BatchNorm2d(num_features=out_chs)),
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
        out = torch.cat([self.gave_pool(x),
                         self.conv1x1(x),
                         self.aspp_bra1(x),
                         self.aspp_bra2(x),
                         self.aspp_bra3(x)], dim=1)

        out = self.aspp_catdown(out)
        return self.upsampling(out)


class ASPPInPlaceABNBlock(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(56, 112),
                 up_ratio=2, aspp_sec=(12, 24, 36), norm_act=ABN):
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


class SDASPPInPlaceABNBlock(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(56, 112),
                 up_ratio=2, aspp_sec=(12, 24, 36), norm_act=ABN):
        super(SDASPPInPlaceABNBlock, self).__init__()

        self.in_norm = norm_act(in_chs)
        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                                                    ("conv1_0", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(size=feat_res, mode='bilinear'))]))

        self.conv1x1 = nn.Sequential(OrderedDict([("conv1_1", nn.Conv2d(in_chs, out_chs, kernel_size=1,
                                                                        stride=1, padding=0, bias=False,
                                                                        groups=1, dilation=1))]))

        self.aspp_bra1 = nn.Sequential(OrderedDict([("dconv2_1", nn.Conv2d(in_chs, in_chs, kernel_size=3,
                                                                           stride=1, padding=aspp_sec[0], bias=False,
                                                                           groups=in_chs, dilation=aspp_sec[0])),
                                                    ("pconv2_1", nn.Conv2d(in_chs, out_chs, kernel_size=1,
                                                                           stride=1, padding=0, bias=False,
                                                                           groups=1, dilation=1))]))

        self.aspp_bra2 = nn.Sequential(OrderedDict([("dconv2_2", nn.Conv2d(in_chs, in_chs, kernel_size=3,
                                                                           stride=1, padding=aspp_sec[1], bias=False,
                                                                           groups=in_chs, dilation=aspp_sec[1])),
                                                    ("pconv2_2", nn.Conv2d(in_chs, out_chs, kernel_size=1,
                                                                           stride=1, padding=0, bias=False,
                                                                           groups=1, dilation=1))]))

        self.aspp_bra3 = nn.Sequential(OrderedDict([("dconv2_3", nn.Conv2d(in_chs, in_chs, kernel_size=3,
                                                                           stride=1, padding=aspp_sec[2], bias=False,
                                                                           groups=in_chs, dilation=aspp_sec[2])),
                                                    ("pconv2_3", nn.Conv2d(in_chs, out_chs, kernel_size=1,
                                                                           stride=1, padding=0, bias=False,
                                                                           groups=1, dilation=1))]))

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

        return self.upsampling(self.aspp_catdown(x))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# For MobileNetV2
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
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
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(inplace=True),

            # dw
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio,
                      kernel_size=3, stride=stride, padding=dilate, dilation=dilate,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(inplace=True),

            # pw-linear
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup,
                      kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
        )

    def forward(self, x):
        if self.use_res_connect:
            return torch.add(x, 1, self.conv(x))
        else:
            return self.conv(x)


class SCSEInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilate, expand_ratio):
        """
        InvertedResidual: Core block of the MobileNetV2
        :param inp:    (int) Number of the input channels
        :param oup:    (int) Number of the output channels
        :param stride: (int) Stride used in the Conv3x3
        :param dilate: (int) Dilation used in the Conv3x3
        :param expand_ratio: (int) Expand ratio of the Channel Width of the Block
        """
        super(SCSEInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels=inp, out_channels=inp * expand_ratio,
                      kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(inplace=True),

            # dw
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=inp * expand_ratio,
                      kernel_size=3, stride=stride, padding=dilate, dilation=dilate,
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(num_features=inp * expand_ratio, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(inplace=True),

            # pw-linear
            nn.Conv2d(in_channels=inp * expand_ratio, out_channels=oup,
                      kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=oup, eps=1e-05, momentum=0.1, affine=True),
            SCSEBlock(channel=oup, reduction=2)
        )

    def forward(self, x):
        if self.use_res_connect:
            return torch.add(x, 1, self.conv(x))
        else:
            return self.conv(x)
