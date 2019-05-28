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
import numpy as np
from torch.autograd import Variable
import functools
import torchvision.models as mo
res18=mo.resnet18()
affine_par = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
# sys.path.append('/inplace_abn')
from utils.resnet_block import conv3x3, Bottleneck,BasicBlock
from utils.psp_block import PSPModule
from utils.modules import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


class Net(nn.Module):
    def __init__(self,student, teacher, criterion_l,criterion_s,criterion_k,args):
        super(Net, self).__init__()
        self.student = student
        self.teacher = teacher
        self.criterion_l = criterion_l
        self.criterion_k = criterion_k
        self.criterion_s = criterion_s
        self.args = args


    def inference(self,image,upsample):
        self.student.eval()
        _, _, w, h = image.size()
        preds = self.student(image)
        if upsample:
            scale_pred = F.upsample(input=preds[0], size=(w, h), mode='bilinear', align_corners=True)
        else:
            scale_pred = preds[0]
        return scale_pred

    def cal_loss(self,s_image,t_image,label):

        preds = self.student(s_image)
        with torch.no_grad():
            soft = self.teacher(t_image)
        loss = 0
        loss_k = 0
        loss_l = 0
        loss_s = 0
        if 'label' in self.args.sd_mode:
            loss_l = self.args.weight_l * self.criterion_l(preds, label)
            loss = loss + loss_l

        if 'kd' in self.args.sd_mode:
            loss_k = self.args.weight_k * self.criterion_k(preds, soft)

            loss = loss + loss_k
        if 'sd' in self.args.sd_mode:
            loss_s = self.args.weight_s * self.criterion_s(preds, soft)
            loss = loss + loss_s
        return loss

    def forward(self,s_image,t_image,label,upsample,mode):
        if mode =='train':
            ret = self.cal_loss(s_image,t_image,label)
        elif mode == 'test':
            ret = self. inference(s_image,upsample)

        return ret
