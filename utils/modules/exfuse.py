import torch.nn as nn
import torch

from collections import OrderedDict


class SemanticSupervision(nn.Module):
    def __init__(self, in_chns, out_chns):
        super(SemanticSupervision, self).__init__()
        self.out_chns = out_chns

        self.semantic = nn.Sequential(OrderedDict([("conv1x7", nn.Conv2d(in_chns, (in_chns // 2) * 3,
                                                                         kernel_size=(1, 7), stride=1,
                                                                         padding=(0, 3), bias=False)),
                                                   ("norm1", nn.BatchNorm2d((in_chns // 2) * 3)),
                                                   ("act1", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
                                                   ("conv7x1", nn.Conv2d((in_chns // 2) * 3, (out_chns // 2) * 3,
                                                                         kernel_size=(7, 1), stride=1,
                                                                         padding=(3, 0), bias=False)),
                                                   ("norm2", nn.BatchNorm2d((out_chns // 2) * 3)),
                                                   ("act2", nn.LeakyReLU(negative_slope=0.1, inplace=True))]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear((out_chns // 2) * 3, out_chns)

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        se = self.semantic(x)

        se = self.avg_pool(se).view(bahs, (self.out_chns // 2) * 3)
        se = self.classifier(se)
        return se


if __name__ == "__main__":
    from torch.autograd import Variable

    dummy_input = Variable(torch.rand(1, 16, 224, 448), requires_grad=True)

    sesuper = SemanticSupervision(in_chns=16, out_chns=19)

    se = sesuper(dummy_input)

    print("ok!!")