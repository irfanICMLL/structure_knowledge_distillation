import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.spectral import SpectralNorm
import numpy as np


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size=64, conv_dim=64,location=0):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        self.location = location
        layer1 = []
        layer2 = []
        layer3 = []
        last = []
        if self.location == 0:

            layer1.append(SpectralNorm(nn.Conv2d(22, conv_dim, 4, 2, 1)))
        else:
            layer1.append(SpectralNorm(nn.Conv2d(4, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2


        layer4 = []
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)
        curr_dim = curr_dim * 2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')



    def forward(self, x,y,gp_mode=False):
        if self.location == 0:
            x = x[0]
            x_p = F.softmax(x, dim=1)
            _,_,w,h=x_p.size()
            y_scale = nn.Upsample(size=(w,h),mode='bilinear',align_corners=True)(y)

            in_ = torch.cat([y_scale,x_p],dim=1)
        else:
            x = x[1].abs_().sum(dim=1,keepdim=True)
            _, _, w, h = x.size()
            y_scale = nn.Upsample(size=(w,h), mode='bilinear', align_corners=True)(y)
            in_ = torch.cat([y_scale,x],dim=1)

        out = self.l1(in_)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)
        if gp_mode==True:
            grad = torch.autograd.grad(outputs=out,
                                       inputs=in_,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
            return d_loss_gp
        else:
            return out.squeeze()

