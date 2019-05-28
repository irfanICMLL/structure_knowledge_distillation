import pdb
import torch.nn as nn
# import encoding.nn as nn
import math
import os
import sys
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from .loss import HNMDiscriminativeLoss, OhemCrossEntropy2d, HardCrossEntropy2d, CrossEntropy2d, \
    OhemCrossEntropy2d_m
from .utils import down_sample_target_count, down_sample_target

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../metric_loss'))


class Hard_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Hard_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  #

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
        attention = self.sigmoid(energy)

        return attention


class CriterionAttnhard(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionAttnhard, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        # self.attn1 = Hard_Attn(2048, 'relu')
        self.attn = Hard_Attn(512, 'relu')
        self.criterion_sd = torch.nn.MSELoss(size_average=True)
        self.criterion_s = torch.nn.CrossEntropyLoss(size_average=True)

    def forward(self, preds, attn_h):
        m_batchsize, C, w, h = preds[2].size()
        graph_s = self.attn(preds[2])
        attn_s = torch.cat((1 - graph_s, graph_s)).view(2, m_batchsize, w * h, w * h).permute(1, 0, 2, 3)
        loss = self.criterion_s(attn_s, attn_h)

        return loss


class CriterionOhemMDSN(nn.Module):
    '''
    DSN + OhemM: consider hard sample mining from multiple levels.
    '''

    def __init__(self, ignore_index=255, thres1=0.8, thres2=0.5, min_kept=0, use_weight=True):
        super(CriterionOhemMDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = OhemCrossEntropy2d_m(ignore_index, thres1, thres2, min_kept, use_weight)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)
        return 0.4 * loss1 + loss2


class Criterion_CrossEntropy_LovaszSoftmax(nn.Module):
    '''
    LovaszSoftmax loss:
        loss functions used to optimize the mIOU directly.
    '''

    def __init__(self, ignore_index=255):
        super(Criterion_CrossEntropy_LovaszSoftmax, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        print("use Cross Entropy + Lovasz Softmax loss")

    def forward(self, preds, target):
        n, h, w = target.size(0), target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corner=True)
        loss = self.criterion(scale_pred, target)

        prob = F.softmax(scale_pred)
        loss_lovasz = lovasz_softmax(prob, target, ignore=self.ignore_index)

        return loss + loss_lovasz


def get_confusion_matrix(gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]
    return confusion_matrix


class CriterionDSNIOU(nn.Module):
    '''
    compute the loss and mIOU for every image based on DSN manner.
    return:
           loss: cross-entropy loss.
           mean_IU: mIOU on every image.
    '''

    def __init__(self, ignore_index=255, num_classes=19):
        super(CriterionDSNIOU, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        n, h, w = target.size(0), target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corner=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corner=True)
        loss2 = self.criterion(scale_pred, target)

        mean_IU = np.zeros(n)
        preds = np.asarray(np.argmax(scale_pred.data.cpu().numpy(), axis=1), dtype=np.uint8)
        labels = np.asarray(target.data.cpu().numpy()[:, :h, :w], dtype=np.int)

        for i in range(n):
            label = labels[i]
            pred = preds[i]
            ignore_index = label != 255
            label = label[ignore_index]
            pred = pred[ignore_index]
            confusion_matrix = get_confusion_matrix(label, pred, self.num_classes)
            pos = confusion_matrix.sum(1)  # col sum
            res = confusion_matrix.sum(0)  # row sum
            tp = np.diag(confusion_matrix)
            IU_array = (tp / np.maximum(1.0, pos + res - tp))
            IU_array = IU_array[IU_array != 0]
            mean_IU[i] = IU_array.mean()

        return 0.4 * loss1 + loss2, mean_IU


class CriterionCrossEntropyIOU(nn.Module):
    '''
    compute the loss and mIOU for every image.
    return:
           loss: cross-entropy loss.
           mean_IU: mIOU on every image.
    '''

    def __init__(self, ignore_index=255, num_classes=19, use_weight=True, use_ohem=False, thres=0.7, min_kept=100000):
        super(CriterionCrossEntropyIOU, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.num_classes = num_classes
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        if use_ohem:
            self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight)

    def forward(self, preds, target):
        n, h, w = target.size(0), target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corner=True)
        # scale_pred_dsn = F.upsample(input=preds[0], size=(h, w),mode='bilinear',align_corner=True)
        mean_IU = np.zeros(n)
        image_weight = np.zeros(n)
        preds_ = np.asarray(np.argmax(scale_pred.data.cpu().numpy(), axis=1), dtype=np.uint8)
        labels = np.asarray(target.data.cpu().numpy()[:, :h, :w], dtype=np.int)

        for i in range(n):
            label = labels[i]
            pred = preds_[i]
            ignore_index = label != 255
            label = label[ignore_index]
            pred = pred[ignore_index]
            confusion_matrix = get_confusion_matrix(label, pred, self.num_classes)
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            IU_array = (tp / np.maximum(1.0, pos + res - tp))
            IU_array = IU_array[IU_array != 0]
            if IU_array.size:
                mean_IU[i] = IU_array.mean()
                image_weight[i] = 1 / mean_IU[i]
            else:
                mean_IU[i] = 0
                image_weight[i] = 0
        # normalize the weights for all images
        image_weight = image_weight / image_weight.sum()

        # loss1 = self.criterion(torch.unsqueeze(scale_pred_dsn[0,:,:], 0), torch.unsqueeze(target[0,:,:], 0))*image_weight[0]
        loss2 = self.criterion(torch.unsqueeze(scale_pred[0, :, :], 0), torch.unsqueeze(target[0, :, :], 0)) * \
                image_weight[0]
        for i in range(1, n):
            # loss1 += self.criterion(torch.unsqueeze(scale_pred_dsn[i,:,:], 0), torch.unsqueeze(target[i,:,:], 0))*image_weight[i]
            loss2 += self.criterion(torch.unsqueeze(scale_pred[i, :, :], 0), torch.unsqueeze(target[i, :, :], 0)) * \
                     image_weight[i]

        # return 0.4*loss1 + loss2
        return loss2


class ImageOHEMCriterionCrossEntropy(nn.Module):
    '''
    Image-level OHEM:
        choose the image with the max loss to compute the gradients.
        The hard sample mining is focused to eliminate the influence of large amount of samples with
        small gradients, thus slow down the model's learning speed.
    '''

    def __init__(self, ignore_index=255):
        super(ImageOHEMCriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        n, h, w = target.size(0), target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corner=True)
        max_loss = 0
        for i in range(n):
            loss = self.criterion(scale_pred[i, :, :], target[i, :, :])
            if loss > max_loss:
                max_loss = loss
        return max_loss


class CriterionCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255, use_weight=False):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        # print('yes')
        loss = self.criterion(scale_pred, target)
        return loss


class OnlineCriterionCrossEntropy(nn.Module):
    '''
    mini-batch-level Class Balance:
        the class unbalance varies according to different images, thus it is better to compute the balance weights according to each mini-batch.
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        # uncomment if use online adaptive class balance.
        pdb.set_trace()
        mask = (target[:, :, :] == 255)
        print('ignore pixel count {}'.format(torch.sum(mask)))
        if self.use_weight:
            freq = np.zeros(19)
            for k in range(19):
                mask = (target[:, :, :] == k)
                freq[k] += torch.sum(mask)
            weight = freq / np.sum(freq)
            self.weight = torch.FloatTensor(weight).cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corner=True)
        loss = self.criterion(scale_pred, target)
        return loss


class CriterionOhemCrossEntropy(nn.Module):
    '''
    Pixel-level OHEM:
        choose the pixels with bigger loss to compute the gradients.
        The hard pixel mining is very unstable and is expensive to find the best hyper-parameters.
    '''

    def __init__(self, ignore_index=255, thres=0.6, min_kept=200000, use_weight=True):
        super(CriterionOhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        # 1/10 of the pixels within a mini-batch, if we use 2x4 on two cards, it should be 200000
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight)

    def forward(self, preds, target):
        # assert len(preds) == 2
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corner=True)
        loss = self.criterion(scale_pred, target)
        # print('OhemCrossEntropy2d Loss: {}'.format(loss.data.cpu().numpy()[0]))
        return loss


class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        if use_weight:
            print("w/ class balance")
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CriterionCenterDSN(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4, center_weight=0.0001):
        super(CriterionCenterDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.center_weight = center_weight
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.criterion_center = HNMDiscriminativeLoss(0.5, 1.5, weight, ignore_index, loss_weight=1)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        scale_target = down_sample_target(target, 8)
        loss3 = self.criterion_center(preds[0], scale_target)

        return self.dsn_weight * loss1 + loss2 + self.center_weight * loss3


class CriterionGloLoc(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionGloLoc, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        # self-attention context features
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        # global average context features
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        # fuse global-average-context with self-attention-context
        scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss3 = self.criterion(scale_pred, target)

        return loss1 + loss2 + loss3


class CriterionGloLocDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4, sa_weight=1, psp_weight=1, fuse_weight=1):
        super(CriterionGloLocDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.psp_weight = psp_weight
        self.sa_weight = sa_weight
        self.fuse_weight = fuse_weight
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        # return [x_dsn, cls_atten, cls_global, cls_fuse]
        h, w = target.size(1), target.size(2)

        # DSN supervision
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss0 = self.criterion(scale_pred, target)

        # self-attention context features
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        # global average context features
        scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        # fuse global-average-context with self-attention-context
        scale_pred = F.upsample(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        loss3 = self.criterion(scale_pred, target)

        return self.dsn_weight * loss0 + self.sa_weight * loss1 + self.psp_weight * loss2 + self.fuse_weight * loss3


class CriterionGloLocSeqDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4):
        super(CriterionGloLocSeqDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        # DSN supervision
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss0 = self.criterion(scale_pred, target)

        # self-attention context features
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        # global average context features
        scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return self.dsn_weight * loss0 + loss1 + loss2


class CriterionDSN_HardImage(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN_HardImage, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        n, h, w = target.size(0), target.size(1), target.size(2)
        pdb.set_trace()
        # scale_pred = F.upsample(input=preds[0], size=(h, w),mode='bilinear',align_corners=True)
        # image_loss = []
        # np_loss  = []
        # for i in range(n):
        #     loss = self.criterion(scale_pred[i,:,:].unsqueeze(0), target[i,:,:].unsqueeze(0))
        #     image_loss.append(loss)
        #     np_loss.append(loss.data.numpy())
        # np_loss = np.asarray(np_loss)
        # index = np.argsort(np_loss)
        # pdb.set_trace()
        # loss1 = torch.sum(torch.topk(image_loss, n/2)[0])

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        image_loss = torch.ones(n)
        pdb.set_trace()
        for i in range(n):
            image_loss[i] = self.criterion(scale_pred[i, :, :].unsqueeze(0), target[i, :, :].unsqueeze(0))
        loss1 = torch.sum(torch.topk(image_loss, n / 2)[0])

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        image_loss = torch.ones(n)
        image_loss = Variable(image_loss, requires_grad=True)
        for i in range(n):
            image_loss[i] = self.criterion(scale_pred[i, :, :].unsqueeze(0), target[i, :, :].unsqueeze(0))
        loss2 = torch.sum(torch.topk(image_loss, n / 2)[0])
        return 0.4 * loss1 + loss2


class Criterion_DSN_LovaszSoftmax(nn.Module):
    '''
    LovaszSoftmax loss:
        loss functions used to optimize the mIOU directly.
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(Criterion_DSN_LovaszSoftmax, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        print("use DSN_LovaszSoftmax loss")

    def forward(self, preds, target):
        n, h, w = target.size(0), target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        prob = F.softmax(scale_pred)
        losslovasz_2 = lovasz_softmax(prob, target, ignore=self.ignore_index)
        loss2 = self.criterion(scale_pred, target)

        return 0.4 * loss1 + (loss2 + losslovasz_2)


class CriterionOhemDSN_LovaszSoftmax(nn.Module):
    '''
    LovaszSoftmax loss:
        loss functions used to optimize the mIOU directly.
    '''

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, use_weight=True):
        super(CriterionOhemDSN_LovaszSoftmax, self).__init__()
        self.ignore_index = ignore_index
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight)
        print("use CriterionOhemDSN_LovaszSoftmax loss")

    def forward(self, preds, target):
        n, h, w = target.size(0), target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        prob = F.softmax(scale_pred)
        losslovasz_1 = lovasz_softmax(prob, target, ignore=self.ignore_index)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        prob = F.softmax(scale_pred)
        losslovasz_2 = lovasz_softmax(prob, target, ignore=self.ignore_index)
        loss2 = self.criterion(scale_pred, target)

        return 0.4 * (loss1 + losslovasz_1) + (loss2 + losslovasz_2)


class CriterionOhemDSN(nn.Module):
    '''
    DSN + Ohem: we find that use hard-mining for both supervision harms the performance.
                Thus we choose the original loss for the shallow supervision
                and the hard-mining loss for the deeper supervision
    '''

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, dsn_weight=0.4, use_weight=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=use_weight)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CriterionOhemDSN_v2(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000, dsn_weight=0.4):
        super(CriterionOhemDSN_v2, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.criterion_ohem = OhemCrossEntropy2d(ignore_index, thres, min_kept, use_weight=True)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion_ohem(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CriterionCrossEntropyMlabel(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropyMlabel, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target, mlabel):
        h, w = target.size(1), target.size(2)
        preds = preds * mlabel.unsqueeze(2).unsqueeze(3).expand(preds.size())
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class CriterionHardCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255, hard_ratio=1):
        super(CriterionHardCrossEntropy, self).__init__()
        self.criterion = HardCrossEntropy2d(ignore_label=ignore_index, hard_ratio=hard_ratio)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class CriterionDSN_Coarse2Fine(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, use_weight=True, dsn_weight=0.4):
        super(CriterionDSN_Coarse2Fine, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss3 = self.criterion(scale_pred, target)

        return self.dsn_weight * loss1 + loss2 + loss3


class CriterionCoarse2FineBase(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCoarse2FineBase, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)

        return loss1 + loss2


class CriterionEMH(nn.Module):
    def __init__(self, ignore_index=255, weights=[1, 1, 1, 1]):
        super(CriterionEMH, self).__init__()
        self.ignore_index = ignore_index
        self.weights = weights
        self.criterion1 = OhemCrossEntropy2d(ignore_index, 0.8, min_kept=40000)
        self.criterion2 = OhemCrossEntropy2d(ignore_index, 0.6, min_kept=20000)
        self.criterion3 = OhemCrossEntropy2d(ignore_index, 0.4, min_kept=10000)
        self.criterion4 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        scale_pred = F.upsample(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        loss3 = self.criterion3(scale_pred, target)

        scale_pred = F.upsample(input=preds[3], size=(h, w), mode='bilinear', align_corners=True)
        loss4 = self.criterion4(scale_pred, target)

        return self.weights[0] * loss1 + self.weights[1] * loss2 + self.weights[2] * loss3 + self.weights[3] * loss4


class CriterionCrossEntropyCoarse2Fine(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropyCoarse2Fine, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        target_2 = down_sample_target(target, 2)
        target_4 = down_sample_target(target, 4)
        target_8 = down_sample_target(target, 8)

        loss = self.criterion(preds[6], target) + self.criterion(preds[7], target)
        loss_2 = self.criterion(preds[4], target_2) + self.criterion(preds[5], target_2)
        loss_4 = self.criterion(preds[2], target_4) + self.criterion(preds[3], target_4)
        loss_8 = self.criterion(preds[0], target_8) + self.criterion(preds[1], target_8)
        print(
            'CrossEntropyLoss 1/8 Loss: {}, 1/4 Loss: {}, 1/2 Loss: {}, 1 Loss: {}'.format(loss_8.data.cpu().numpy()[0], \
                                                                                           loss_4.data.cpu().numpy()[0],
                                                                                           loss_2.data.cpu().numpy()[0],
                                                                                           loss.data.cpu().numpy()[0]))
        return loss + 0.5 * loss_2 + 0.25 * loss_4 + 0.125 * loss_8


class CriterionCrossEntropyDUC(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropyDUC, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        # scale_pred = F.upsample(input=preds[0], size=(h, w),mode='bilinear',align_corners=True)
        # scale_pred = F.upsample(input=preds, size=(h, w),mode='bilinear',align_corners=True)
        loss = self.criterion(preds, target)
        print('CrossEntropyLoss Loss: {}'.format(loss.data.cpu().numpy()[0]))
        return loss


# class CriterionCenterDSN(nn.Module):
#     def __init__(self, num_classes=19, ignore_index=255, center_weight=0.001, dsn_weight=0.4):
#         super(CriterionCenterDSN, self).__init__()
#         self.center_weight = center_weight
#         self.dsn_weight = dsn_weight
#         self.ignore_index = ignore_index
#         weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, \
#             1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
#         self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
#         self.criterion_center = CenterLoss(num_classes, feature_dim=512, ignore_label=ignore_index)

#     def forward(self, preds, target):
#         h, w = target.size(1), target.size(2)

#         scale_pred = F.upsample(input=preds[1], size=(h, w),mode='bilinear',align_corners=True)
#         loss1 = self.criterion(scale_pred, target)

#         scale_pred = F.upsample(input=preds[2], size=(h, w),mode='bilinear',align_corners=True)
#         loss2 = self.criterion(scale_pred, target)

#         scale_target = down_sample_target(target, 8)
#         loss3 = self.criterion_center(preds[0], scale_target)

#         return self.dsn_weight*loss1 + loss2 + self.center_weight*loss3


class CriterionDownSampleTarget(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionDownSampleTarget, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = OhemCrossEntropy2d(ignore_index, 0.7, 100000)
        self.criterion2 = HNMDiscriminativeLoss(0.2, 1.5, ignore_index, loss_weight=0.1)

    def forward(self, preds, target):
        assert len(preds) == 2
        h, w = target.size(1), target.size(2)
        _, _, feature_h, feature_w = preds[0].size()

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        # scale_target = down_sample_target(target, input_scale=feature_h, output_scale=h)
        scale_target = down_sample_target_count(target, 2)
        loss2 = self.criterion2(preds[1], scale_target)

        print('OhemCrossEntropy2d Loss: {}, Down-Sample HNMDiscriminative Loss : {}'.format(loss1.data.cpu().numpy()[0],
                                                                                            loss2.data.cpu().numpy()[
                                                                                                0]))
        return loss1 + loss2


class CriterionDis(nn.Module):
    def __init__(self, ignore_index=255, alpha=1):
        super(CriterionDis, self).__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        weight = torch.FloatTensor(
            [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
             0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.criterion2 = HNMDiscriminativeLoss(0.5, 1.5, ignore_index, loss_weight=1)

    def forward(self, preds, target):
        # assert len(preds) == 2
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        print(loss1.data.cpu().numpy()[0], loss2.data.cpu().numpy()[0])
        print('CrossEntropy2d Loss: {}, HNMDiscriminative Loss : {}'.format(loss1.data.cpu().numpy()[0],
                                                                            loss2.data.cpu().numpy()[0]))
        return loss1 + self.alpha * loss2


class CriterionHistogram(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionHistogram, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = OhemCrossEntropy2d(ignore_index, 0.7, 100000)
        self.criterion2 = HistogramLoss(num_steps=150, use_gpu=True)

    def forward(self, preds, target):
        assert len(preds) == 2
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        print('OhemCrossEntropy2d Loss: {}, Histogram Loss : {}'.format(loss1.data.cpu().numpy()[0],
                                                                        loss2.data.cpu().numpy()[0]))
        return loss1 + loss2


#####sd_loss
# alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
# interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
# out, _, _ = Dnet(interpolated)
#
# grad = torch.autograd.grad(outputs=out,
#                            inputs=interpolated,
#                            grad_outputs=torch.ones(out.size()).cuda(),
#                            retain_graph=True,
#                            create_graph=True,
#                            only_inputs=True)[0]
#
# grad = grad.view(grad.size(0), -1)
# grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
# d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
#
# # Backward + Optimize
# d_loss = 10 * d_loss_gp


class CriterionFeaSum(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionFeaSum, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
                 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        # self.attn1 = Cos_Attn(2048, 'relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss(size_average=True)

    def forward(self, preds, soft):
        cs, ct = preds[1].size(1), soft[1].size(1)
        graph_s = torch.sum(torch.abs(preds[1]), dim=1, keepdim=True) / cs
        graph_t = torch.sum(torch.abs(soft[1]), dim=1, keepdim=True) / ct
        loss_graph = self.criterion_sd(graph_s, graph_t)
        return loss_graph
        # torch.abs


class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, ignore_index=255, upsample=False, use_weight=True, T=1, sp=0, pp=0):
        super(CriterionKD, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.upsample = upsample
        self.soft_p = sp
        self.pred_p = pp
        self.T = T
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, preds, soft):
        h, w = soft[self.soft_p].size(2), soft[self.soft_p].size(3)
        if self.upsample:
            scale_pred = F.upsample(input=preds[self.pred_p], size=(h * 8, w * 8), mode='bilinear', align_corners=True)
        else:
            scale_pred = preds[self.pred_p]
        scale_soft = F.upsample(input=soft[self.soft_p], size=(h * 8, w * 8), mode='bilinear', align_corners=True)
        loss2 = self.criterion_kd(F.log_softmax(scale_pred / self.T, dim=1), F.softmax(scale_soft / self.T, dim=1))
        return loss2


class Cos_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
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
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / nm
        attention = self.softmax(norm_energy)  # BX (N) X (N)
        return attention


class Cos_Attn_sig(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn_sig, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Sigmoid()  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / nm
        attention = self.softmax(norm_energy)  # BX (N) X (N)
        return attention


class Cos_Attn_no(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn_no, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Sigmoid()  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / nm
        # attention = self.softmax(norm_energy)  # BX (N) X (N)
        return norm_energy


class CriterionLS(nn.Module):
    '''
       local structure with L2 norm
    '''

    def __init__(self, ignore_index=255, location=0, use_weight=True):
        super(CriterionLS, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.location = location
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    # def cosistance(self, logits):
    #     # b, c, w, h = logits.shape
    #     # cosis = torch.zeros(b,1,w-2,h-2)
    #     # for bs in range(b):
    #     #     for i in range(1,w-1):
    #     #         for j in range(1,h-1):
    #     #             x = logits[bs,:,i,j]
    #     #             dis1 = torch.norm(x-logits[bs,:,i-1,j])**2+torch.norm(x-logits[bs,:,i,j-1])**2+torch.norm(x-logits[bs,:,i-1,j-1])**2+torch.norm(x-logits[bs,:,i+1,j])**2
    #     #             dis2 = torch.norm(x-logits[bs,:,i,j+1])**2+torch.norm(x-logits[bs,:,i-1,j+1])**2+torch.norm(x-logits[bs,:,i+1,j+1])**2+torch.norm(x-logits[bs,:,i+1,j-1])**2
    #     #             cosis[bs,0,i-1,j-1] = dis1+dis2
    #     b, c, w, h = logits.shape
    #     x = torch.zeros(b, c, w + 2, h + 2)
    #     x[:, :, 1:w + 1, 1:h + 1] = logits
    #     loss_temp = 0
    #     for i in range(0, 1, 2):
    #         for j in range(0, 1, 2):
    #             if i == 1 and j == 1:
    #                 continue
    #             else:
    #                 temp = torch.zeros(b, c, w + 2, h + 2)
    #                 temp[:, :, i:w + i, j:h + j] = logits
    #                 loss_temp = loss_temp + torch.norm(x - temp) ** 2
    #     return loss_temp

    # def stupid(self, logits):
    #     b, c, w, h = logits.shape
    #     cosis = torch.zeros(b,1,w-2,h-2)
    #     for bs in range(b):
    #         for i in range(1,w-1):
    #             for j in range(1,h-1):
    #                 x = logits[bs,:,i,j]
    #                 dis1 = torch.norm(x-logits[bs,:,i-1,j])**2+torch.norm(x-logits[bs,:,i,j-1])**2+torch.norm(x-logits[bs,:,i-1,j-1])**2+torch.norm(x-logits[bs,:,i+1,j])**2
    #                 dis2 = torch.norm(x-logits[bs,:,i,j+1])**2+torch.norm(x-logits[bs,:,i-1,j+1])**2+torch.norm(x-logits[bs,:,i+1,j+1])**2+torch.norm(x-logits[bs,:,i+1,j-1])**2
    #                 cosis[bs,0,i-1,j-1] = dis1+dis2
    #     # b, c, w, h = logits.shape
    #     # x = torch.zeros(b, c, w + 2, h + 2)
    #     # x[:, :, 1:w + 1, 1:h + 1] = logits
    #     # loss_temp = 0
    #     # for i in range(0, 1, 2):
    #     #     for j in range(0, 1, 2):
    #     #         if i == 1 and j == 1:
    #     #             continue
    #     #         else:
    #     #             temp = torch.zeros(b, c, w + 2, h + 2)
    #     #             temp[:, :, i:w + i, j:h + j] = logits
    #     #             loss_temp = loss_temp + torch.norm(x - temp) ** 2
    #     return cosis

    def f(self, logit):
        b, c, w, h = logit.shape
        filter_181 = torch.from_numpy(np.array([[1, 1, 1], [1, 8, 1], [1, 1, 1]])).expand(1, c, 3, 3).float().cuda()
        filter_101 = torch.from_numpy(np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])).expand(c, 1, 3, 3).float().cuda()
        filter_010 = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])).expand(c, 1, 3, 3).float().cuda()

        aaa = F.conv2d(logit ** 2, filter_181)
        bbb = F.conv2d(logit, filter_101, groups=c)
        ccc = F.conv2d(logit, filter_010, groups=c)
        map = aaa - 2 * torch.sum(bbb * ccc, 1, keepdim=True)
        map.cuda()
        return map

    def forward(self, preds, soft):
        graph_s = self.f(preds[1])
        graph_t = self.f(soft[1])
        loss_graph = self.criterion_sd(graph_s, graph_t)
        return loss_graph


class CriterionLSCos(nn.Module):
    '''
    local structure with cosine similarity
    '''

    def __init__(self, ignore_index=255, location=0, use_weight=True):
        super(CriterionLSCos, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.location = location
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    # def cosistance(self, logits):
    #     # b, c, w, h = logits.shape
    #     # cosis = torch.zeros(b,1,w-2,h-2)
    #     # for bs in range(b):
    #     #     for i in range(1,w-1):
    #     #         for j in range(1,h-1):
    #     #             x = logits[bs,:,i,j]
    #     #             dis1 = torch.norm(x-logits[bs,:,i-1,j])**2+torch.norm(x-logits[bs,:,i,j-1])**2+torch.norm(x-logits[bs,:,i-1,j-1])**2+torch.norm(x-logits[bs,:,i+1,j])**2
    #     #             dis2 = torch.norm(x-logits[bs,:,i,j+1])**2+torch.norm(x-logits[bs,:,i-1,j+1])**2+torch.norm(x-logits[bs,:,i+1,j+1])**2+torch.norm(x-logits[bs,:,i+1,j-1])**2
    #     #             cosis[bs,0,i-1,j-1] = dis1+dis2
    #     b, c, w, h = logits.shape
    #     x = torch.zeros(b, c, w + 2, h + 2)
    #     x[:, :, 1:w + 1, 1:h + 1] = logits
    #     loss_temp = 0
    #     for i in range(0, 1, 2):
    #         for j in range(0, 1, 2):
    #             if i == 1 and j == 1:
    #                 continue
    #             else:
    #                 temp = torch.zeros(b, c, w + 2, h + 2)
    #                 temp[:, :, i:w + i, j:h + j] = logits
    #                 loss_temp = loss_temp + torch.norm(x - temp) ** 2
    #     return loss_temp

    # def stupid(self, logits):
    #     b, c, w, h = logits.shape
    #     cosis = torch.zeros(b,1,w-2,h-2)
    #     for bs in range(b):
    #         for i in range(1,w-1):
    #             for j in range(1,h-1):
    #                 x = logits[bs,:,i,j]
    #                 dis1 = torch.norm(x-logits[bs,:,i-1,j])**2+torch.norm(x-logits[bs,:,i,j-1])**2+torch.norm(x-logits[bs,:,i-1,j-1])**2+torch.norm(x-logits[bs,:,i+1,j])**2
    #                 dis2 = torch.norm(x-logits[bs,:,i,j+1])**2+torch.norm(x-logits[bs,:,i-1,j+1])**2+torch.norm(x-logits[bs,:,i+1,j+1])**2+torch.norm(x-logits[bs,:,i+1,j-1])**2
    #                 cosis[bs,0,i-1,j-1] = dis1+dis2
    #     # b, c, w, h = logits.shape
    #     # x = torch.zeros(b, c, w + 2, h + 2)
    #     # x[:, :, 1:w + 1, 1:h + 1] = logits
    #     # loss_temp = 0
    #     # for i in range(0, 1, 2):
    #     #     for j in range(0, 1, 2):
    #     #         if i == 1 and j == 1:
    #     #             continue
    #     #         else:
    #     #             temp = torch.zeros(b, c, w + 2, h + 2)
    #     #             temp[:, :, i:w + i, j:h + j] = logits
    #     #             loss_temp = loss_temp + torch.norm(x - temp) ** 2
    #     return cosis

    def f(self, logit):
        b, c, w, h = logit.shape
        filter_11 = torch.from_numpy(np.array( [[[1, 0, 0], [0, 1, 0], [0, 0, 0]],[[0, 1, 0], [0, 1, 0], [0, 0, 0]],[[0, 0, 1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [1, 1, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1, 1], [0, 0, 0]],[[0, 0, 0], [0, 1, 0], [1, 0, 0]],[[0, 0, 0], [0, 1, 0], [0, 1, 0]],[[0, 0, 0], [0, 1, 0], [0, 0, 1]]])).expand(c,8, 3, 3).float().cuda()
        filter_11 = torch.transpose(filter_11,0,1)

        filter_10 = torch.from_numpy(np.array( [[[1, 0, 0], [0, 0, 0], [0, 0, 0]],[[0, 1, 0], [0, 0, 0], [0, 0, 0]],[[0, 0, 1], [0, 0, 0], [0, 0, 0]],[[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 1], [0, 0, 0]],[[0, 0, 0], [0, 0, 0], [1, 0, 0]],[[0, 0, 0], [0, 0, 0], [0, 1, 0]],[[0, 0, 0], [0, 0, 0], [0, 0, 1]]])).expand(c,8, 3, 3).float().cuda()
        filter_10 = torch.transpose(filter_10,0,1)

        filter_01 = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])).expand(8,c, 3, 3).float().cuda()


        # filter_11_dep = torch.from_numpy(np.array( [[[1, 0, 0], [0, 1, 0], [0, 0, 0]],[[0, 1, 0], [0, 1, 0], [0, 0, 0]],[[0, 0, 1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [1, 1, 0], [0, 0, 0]],
        #             [[0, 0, 0], [0, 1, 1], [0, 0, 0]],[[0, 0, 0], [0, 1, 0], [1, 0, 0]],[[0, 0, 0], [0, 1, 0], [0, 1, 0]],[[0, 0, 0], [0, 1, 0], [0, 0, 1]]])).expand(c, 8, 3, 3).float().cuda()
        filter_11_dep=torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])).expand(c,1, 3, 3).float().cuda()
        xxaa = F.conv2d(logit ** 2, filter_11)
        aa = F.conv2d(logit ** 2, filter_10)
        xx = F.conv2d(logit ** 2, filter_01)
        kernel_set = np.array( [[[0, 1, 0], [0, 1, 0], [0, 0, 0]],[[0, 0, 1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [1, 1, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 1, 1], [0, 0, 0]],[[0, 0, 0], [0, 1, 0], [1, 0, 0]],[[0, 0, 0], [0, 1, 0], [0, 1, 0]],[[0, 0, 0], [0, 1, 0], [0, 0, 1]]])

        filter_11_dep = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])).expand(c, 1, 3, 3).float().cuda()
        xa = F.conv2d(logit, filter_11_dep, groups=c)
        xa2 = torch.sum(xa * xa, 1, keepdim=True)

        for kernel_ in kernel_set:
            filter_11_dep = torch.from_numpy(kernel_).expand(c, 1, 3,3).float().cuda()
            xa = F.conv2d(logit, filter_11_dep, groups=c)

            tem = torch.sum(xa * xa, 1, keepdim=True)
            xa2 = torch.cat((xa2,tem),dim=1)
        map = (xa2-xxaa)/2*torch.rsqrt(xx)*torch.rsqrt(aa)
        map.cuda()
        return map



    def forward(self, preds, soft):
        graph_s = self.f(preds[1])
        graph_t = self.f(soft[1])
        loss_graph = self.criterion_sd(graph_s, graph_t)
        return loss_graph

class CriterionSDcos_sig(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionSDcos_sig, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.attn = Cos_Attn_sig('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds[1])
        graph_t = self.attn(soft[1])
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph


class CriterionSDcos(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True, pp=1, sp=1):
        super(CriterionSDcos, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.soft_p = sp
        self.pred_p = pp
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.attn = Cos_Attn('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds[self.pred_p])
        graph_t = self.attn(soft[self.soft_p])
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph


class Criterion_CrossEntropy_maoke(nn.Module):
    '''
        loss functions used to optimize the directly.
    '''

    def __init__(self, ignore_index=255, weight=True):
        super(Criterion_CrossEntropy_maoke, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        # self.criterion = F.nll_loss(?)
        # print("use Lovasz Softmax loss")

    def forward(self, preds, target):
        n, h, w = target.size(0), target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        n, c, h, w = scale_pred.size()  # input: (n, c, h, w), target: (n, h, w)
        log_p = F.log_softmax(scale_pred)  # log_p: (n, c, h, w)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)  # log_p: (n*h*w, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) < 20]  # target: (n*h*w,)
        log_p = log_p.view(-1, c)
        mask = target < 20
        target = target[mask]
        CLASS_WEIGHT = [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                        0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365]
        CLASS_WEIGHT = torch.FloatTensor(CLASS_WEIGHT).cuda()
        if self.weight:
            loss = F.nll_loss(log_p, target, weight=CLASS_WEIGHT, size_average=False)
        else:
            loss = F.nll_loss(log_p, target, weight=None, size_average=False)
        loss = loss.float() / mask.data.sum().float()
        return loss


class CriterionSDcos_no(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionSDcos_no, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.attn = Cos_Attn_no('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds[1])
        graph_t = self.attn(soft[1])
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph


class CriterionSDcos_no_sp(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionSDcos_no_sp, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.attn = Cos_Attn_no('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds[0])
        graph_t = self.attn(soft[0])
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph


class CriterionCrossEntropyo(nn.Module):
    def __init__(self, upsample=False, use_weight=True, ignore_index=255):
        super(CriterionCrossEntropyo, self).__init__()
        self.ignore_index = ignore_index
        self.upsample = upsample
        self.use_weight = use_weight
        if use_weight:
            weight = torch.FloatTensor(
                [0.8194, 0.8946, 0.9416, 1.0091, 0.9925, 0.9740, 1.0804, 1.0192, 0.8528,
                 0.9771, 0.9139, 0.9744, 1.1098, 0.8883, 1.0639, 1.2476, 1.0729, 1.1323, 1.0365])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self.upsample:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = preds[0]
        # scale_pred = F.upsample(input=preds, size=(h, w),mode='bilinear',align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss
