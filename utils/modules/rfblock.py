import time
import torch
import torch.nn as nn

from modules import InPlaceABN
from collections import OrderedDict
from torch.autograd import Variable


class RFBlock(nn.Module):
    def __init__(self, in_chs, out_chs, scale=0.1, feat_res=(56, 112), aspp_sec=(12, 24, 36),
                 up_ratio=2, norm_act=InPlaceABN):
        super(RFBlock, self).__init__()
        self.scale = scale

        self.down_chs = nn.Sequential(OrderedDict([("norm_act", norm_act(in_chs)),
                                                   ("down_conv1x1", nn.Conv2d(in_chs, out_chs,
                                                                              kernel_size=1, stride=1,
                                                                              padding=0, bias=False))]))

        self.gave_pool = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                    ("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                                                    ("conv1_0", nn.Conv2d(out_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(size=feat_res, mode='bilinear'))]))

        self.branch0 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                  ("conv1x1", nn.Conv2d(out_chs, out_chs,
                                                                        kernel_size=1, stride=1,
                                                                        padding=0, bias=False)),
                                                  ("norm_act", norm_act(out_chs)),
                                                  ("aconv1", nn.Conv2d(out_chs, out_chs,
                                                                       kernel_size=3, stride=1,
                                                                       padding=1, dilation=1,
                                                                       bias=False))]))

        self.branch1 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                  ("conv1x3", nn.Conv2d(out_chs, (out_chs // 2) * 3,
                                                                        kernel_size=(1, 3), stride=1,
                                                                        padding=(0, 1), bias=False)),
                                                  ("norm_act", norm_act((out_chs // 2) * 3)),
                                                  ("conv3x1", nn.Conv2d((out_chs // 2) * 3, out_chs,
                                                                        kernel_size=(3, 1), stride=1,
                                                                        padding=(1, 0), bias=False)),
                                                  ("norm_act", norm_act(out_chs)),
                                                  ("aconv3", nn.Conv2d(out_chs, out_chs,
                                                                       kernel_size=3, stride=1,
                                                                       padding=aspp_sec[0],
                                                                       dilation=aspp_sec[0],
                                                                       bias=False))]))

        self.branch2 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                  ("conv1x5", nn.Conv2d(out_chs, (out_chs // 2) * 3,
                                                                        kernel_size=(1, 5), stride=1,
                                                                        padding=(0, 2), bias=False)),
                                                  ("norm_act", norm_act((out_chs // 2) * 3)),
                                                  ("conv5x1", nn.Conv2d((out_chs // 2) * 3, out_chs,
                                                                        kernel_size=(5, 1), stride=1,
                                                                        padding=(2, 0), bias=False)),
                                                  ("norm_act", norm_act(out_chs)),
                                                  ("aconv5", nn.Conv2d(out_chs, out_chs,
                                                                       kernel_size=3, stride=1,
                                                                       padding=aspp_sec[1],
                                                                       dilation=aspp_sec[1],
                                                                       bias=False))]))

        self.branch3 = nn.Sequential(OrderedDict([("norm_act", norm_act(out_chs)),
                                                  ("conv1x7", nn.Conv2d(out_chs, (out_chs // 2) * 3,
                                                                        kernel_size=(1, 7), stride=1,
                                                                        padding=(0, 3), bias=False)),
                                                  ("norm_act", norm_act((out_chs // 2) * 3)),
                                                  ("conv7x1", nn.Conv2d((out_chs // 2) * 3, out_chs,
                                                                        kernel_size=(7, 1), stride=1,
                                                                        padding=(3, 0), bias=False)),
                                                  ("norm_act", norm_act(out_chs)),
                                                  ("aconv7", nn.Conv2d(out_chs, out_chs,
                                                                       kernel_size=3, stride=1,
                                                                       padding=aspp_sec[2],
                                                                       dilation=aspp_sec[2],
                                                                       bias=False))]))

        self.conv_linear = nn.Sequential(OrderedDict([("conv1x1_linear", nn.Conv2d(out_chs * 5, out_chs,
                                                                                   kernel_size=1, stride=1,
                                                                                   padding=0, bias=False))]))

        self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio),
                                            int(feat_res[1] * up_ratio)),
                                      mode='bilinear')

    def forward(self, x):
        down = self.down_chs(x)
        out = torch.cat([self.gave_pool(down.clone()),
                         self.branch0(down.clone()),
                         self.branch1(down.clone()),
                         self.branch2(down.clone()),
                         self.branch3(down.clone())], dim=1)

        return self.upsampling(torch.add(self.conv_linear(out), self.scale, down))  # out=input+value×other


if __name__ == "__main__":
    from functools import partial
    from modules import InPlaceABNWrapper
    input_chs = 712
    output_chs = 256
    feat_maps = Variable(torch.randn(1, input_chs, 32, 32).cuda())

    rfblocka = RFBlock(in_chs=input_chs, out_chs=output_chs,
                       scale=0.1, feat_res=(32, 32),
                       norm_act=partial(InPlaceABNWrapper, activation="leaky_relu", slope=0.1)).cuda()

    start_time = time.time()
    _ = rfblocka(feat_maps)
    end_time = time.time()
    print("RFBlock: {}s".format(end_time - start_time))
