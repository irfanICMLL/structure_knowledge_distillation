import torch
import torch.nn as nn

from .bn import ABN
from .misc import SEBlock
from collections import OrderedDict


class DualPathInPlaceABNBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc,
                 groups=1, dilation=1, block_type='normal', norm_act=ABN):
        super(DualPathInPlaceABNBlock, self).__init__()

        self.num_1x1_c = num_1x1_c
        self.dilation = dilation
        self.groups = groups
        self.inc = inc

        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = nn.Sequential(OrderedDict([("conv1x1_w_s2_bn", norm_act(in_chs)),
                                                            ("conv1x1_w_s2", nn.Conv2d(in_chs, num_1x1_c + 2 * inc,
                                                             kernel_size=1, stride=2, padding=0, groups=self.groups,
                                                             dilation=1, bias=False))]))
            else:
                self.c1x1_w_s1 = nn.Sequential(OrderedDict([("conv1x1_w_s1_bn", norm_act(in_chs)),
                                                            ("conv1x1_w_s1", nn.Conv2d(in_chs, num_1x1_c + 2 * inc,
                                                             kernel_size=1, stride=1, padding=0, groups=self.groups,
                                                             dilation=1, bias=False))]))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. 1x1 group point-wise convolution
        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        self.c1x1_a = nn.Sequential(OrderedDict([("conv1x1_a_bn", norm_act(in_chs)),
                                                 ("conv1x1_a", nn.Conv2d(in_chs, num_1x1_a,
                                                  kernel_size=1, stride=1, padding=0, groups=self.groups,
                                                  dilation=1, bias=False))]))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. 3x3 depth-wise convolution
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        if self.dilation > 1 and self.key_stride == 1:
            self.c3x3_b = nn.Sequential(OrderedDict([("conv3x3_b_bn", norm_act(num_1x1_a)),
                                                     ("conv3x3_b", nn.Conv2d(num_1x1_a, num_3x3_b,
                                                      kernel_size=3, stride=1, padding=dilation,
                                                      groups=num_1x1_a, dilation=self.dilation, bias=False))]))
        else:
            self.c3x3_b = nn.Sequential(OrderedDict([("conv3x3_b_bn", norm_act(num_1x1_a)),
                                                     ("conv3x3_b", nn.Conv2d(num_1x1_a, num_3x3_b,
                                                      kernel_size=3, stride=self.key_stride, padding=1,
                                                      groups=num_1x1_a, dilation=1, bias=False))]))

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. 1x1 group point-wise convolution
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        self.c1x1_c = nn.Sequential(OrderedDict([("conv1x1_c_bn", norm_act(num_3x3_b)),
                                                 ("conv1x1_c", nn.Conv2d(num_3x3_b, num_1x1_c + inc,
                                                  kernel_size=1, stride=1, padding=0,
                                                  groups=self.groups, dilation=1, bias=False)),
                                                 ("se_block", SEBlock(num_1x1_c + inc, 16)),
                                                 ("dropout", nn.Dropout2d(p=0.2, inplace=True))]))

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
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x

        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in.clone())
            else:
                x_s = self.c1x1_w_s1(x_in.clone())

            x_s = self._channel_shuffle(x_s, self.groups)          # shuffle channels in group
            x_s = torch.split(x_s, self.num_1x1_c, dim=1)    # split channels for res and dense
        else:
            x_s = x

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. 1x1 group point-wise convolution
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        x_r = self.c1x1_a(x_in)
        x_r = self._channel_shuffle(x_r, self.groups)  # shuffle channels in group

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. 3x3 depth-wise convolution
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        x_r = self.c3x3_b(x_r)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        # 3. 1x1 group point-wise convolution
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
        x_r = self.c1x1_c(x_r)
        x_r = self._channel_shuffle(x_r, self.groups)  # shuffle channels in group
        x_r = torch.split(x_r, self.num_1x1_c, dim=1)  # split channels for res and dense

        resid = torch.add(x_s[0], 1, x_r[0])         # for res-net
        dense = torch.cat([x_s[1], x_r[1]], dim=1)   # for dense-net
        return resid, dense

