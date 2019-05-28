from modules import InPlaceABN, ASPPInPlaceABNBlock, InPlaceABNWrapper

import torch.nn as nn
import encoding
import torch


class ContextEncodeInplaceABN(nn.Module):
    def __init__(self, channel, K=16, reduction=4, norm_act=InPlaceABN):
        super(ContextEncodeInplaceABN, self).__init__()
        out_channel = int(channel / reduction)

        self.pre_abn = norm_act(channel)
        self.context_enc = nn.Sequential(norm_act(channel),
                                         nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, padding=0),
                                         norm_act(out_channel),
                                         encoding.nn.Encoding(D=out_channel, K=K),
                                         encoding.nn.View(-1, out_channel*K),
                                         encoding.nn.Normalize())

        self.channel_se = nn.Sequential(nn.Linear(out_channel * K, channel), nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()

        pre_x = self.pre_abn(x.clone())
        encode = self.context_enc(pre_x)
        chn_se = self.channel_se(encode).view(batch_size, num_channels, 1, 1)

        spa_se = self.spatial_se(pre_x)

        return encode, torch.mul(torch.mul(x, spa_se), chn_se)


class ContextEncodeDropInplaceABN(nn.Module):
    def __init__(self, channel, K=16, reduction=4, norm_act=InPlaceABN):
        super(ContextEncodeDropInplaceABN, self).__init__()
        out_channel = int(channel / reduction)

        self.pre_abn = norm_act(channel)
        self.context_enc = nn.Sequential(nn.Conv2d(channel, out_channel, kernel_size=1,
                                                   stride=1, padding=0),
                                         norm_act(out_channel),
                                         encoding.nn.EncodingDrop(D=out_channel, K=K),
                                         encoding.nn.View(-1, out_channel*K),
                                         encoding.nn.Normalize())

        self.channel_se = nn.Sequential(nn.Linear(out_channel*K, channel), nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()

        pre_x = self.pre_abn(x.clone())
        encode = self.context_enc(pre_x)
        chn_se = self.channel_se(encode).view(batch_size, num_channels, 1, 1)

        spa_se = self.spatial_se(pre_x)

        return encode, torch.mul(torch.mul(x, spa_se), chn_se)


if __name__ == "__main__":
    from functools import partial
    from torch.autograd import Variable

    B, C, H, W, K = 2, 32, 56, 56, 32
    dummy_in = Variable(torch.randn(B, C, H, W).cuda(), requires_grad=True)

    context = ContextEncodeInplaceABN(channel=C, K=K, reduction=4,
                                      norm_act=partial(InPlaceABNWrapper,
                                                       activation="leaky_relu",
                                                       slope=0.1)).cuda()

    enc, scored = context(dummy_in)
    print("ok!!!")

    context_drop = ContextEncodeDropInplaceABN(channel=C, K=K, reduction=4,
                                               norm_act=partial(InPlaceABNWrapper,
                                                                activation="leaky_relu",
                                                                slope=0.1)).cuda()

    enc, scored = context(dummy_in)
    print("ok!!!")
