import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2


class OhemCrossEntropy2d_m(nn.Module):
    def __init__(self, ignore_label=255, thresh1=0.8, thresh2=0.5, min_kept=0, use_weight=True):
        super(OhemCrossEntropy2d_m, self).__init__()
        self.ignore_label = ignore_label
        self.thresh1 = float(thresh1)
        self.thresh2 = float(thresh2)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
                 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            print('OhemCrossEntropy2d weights : {}'.format(weight))
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]

            kept_easy = pred >= self.thresh1
            easy_inds = valid_inds[kept_easy]

            kept_middle = (pred < self.thresh1) & (pred >= self.thresh2)
            middle_inds = valid_inds[kept_middle]

            kept_hard = pred < self.thresh2
            hard_inds = valid_inds[kept_hard]

            # print('hard ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))

        # generate the indexes for easy samples
        easy_label = input_label.copy()
        valid_inds = easy_inds
        label = easy_label[valid_inds].copy()
        easy_label.fill(self.ignore_label)
        easy_label[valid_inds] = label
        valid_flag_new = easy_label != self.ignore_label
        target = Variable(torch.from_numpy(easy_label.reshape(target.size())).long().cuda())
        loss1 = self.criterion(predict, target)
        # print('easy ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))
        # generate the indexes for middle-hard samples
        middle_label = input_label.copy()
        valid_inds = middle_inds
        label = middle_label[valid_inds].copy()
        middle_label.fill(self.ignore_label)
        middle_label[valid_inds] = label
        valid_flag_new = middle_label != self.ignore_label
        target = Variable(torch.from_numpy(middle_label.reshape(target.size())).long().cuda())
        loss2 = self.criterion(predict, target)
        # print('middle ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))
        # generate the indexes for hard samples
        hard_label = input_label.copy()
        valid_inds = hard_inds
        label = hard_label[valid_inds].copy()
        hard_label.fill(self.ignore_label)
        hard_label[valid_inds] = label
        valid_flag_new = hard_label != self.ignore_label
        target = Variable(torch.from_numpy(hard_label.reshape(target.size())).long().cuda())
        loss3 = self.criterion(predict, target)
        # print('hard ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))
        return loss1 + loss2 + loss3


class HardCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, hard_ratio=1):
        super(HardCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.hard_ratio = hard_ratio

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        num_keep = int(num_valid * self.hard_ratio)

        prob = input_prob[:, valid_flag]
        pred = prob[label, np.arange(len(label), dtype=np.int32)]

        index = (-pred).argsort()
        threshold_index = index[num_keep - 1]
        threshold = pred[threshold_index]
        kept_flag = pred <= threshold
        valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, use_weight=True):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.use_weight = use_weight
        if self.use_weight:
            self.weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
                 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]).cuda()
            print('CrossEntropy2d weights : {}'.format(self.weight))
        else:
            self.weight = None

    def forward(self, predict, target, weight=None):

        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # Variable(torch.randn(2,10)
        if self.use_weight:
            print('target size {}'.format(target.shape))
            freq = np.zeros(19)
            for k in range(19):
                mask = (target[:, :, :] == k)
                freq[k] = torch.sum(mask)
                print('{}th frequency {}'.format(k, freq[k]))
            weight = freq / np.sum(freq)
            print(weight)
            self.weight = torch.FloatTensor(weight)
            print('Online class weight: {}'.format(self.weight))
        else:
            self.weight = None

        criterion = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_label)
        # torch.FloatTensor([2.87, 13.19, 5.11, 37.98, 35.14, 30.9, 26.23, 40.24, 6.66, 32.07, 21.08, 28.14, 46.01, 10.35, 44.25, 44.9, 44.25, 47.87, 40.39])
        # weight = Variable(torch.FloatTensor([1, 1.49, 1.28, 1.62, 1.62, 1.62, 1.64, 1.62, 1.49, 1.62, 1.43, 1.62, 1.64, 1.43, 1.64, 1.64, 1.64, 1.64, 1.62]), requires_grad=False).cuda()
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = criterion(predict, target)
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.6, min_kept=0, use_weight=True):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
                 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            # print('hard ratio: {} = {} / {} '.format(round(len(valid_inds)/num_valid, 4), len(valid_inds), num_valid))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class DiscriminativeLoss(nn.Module):

    def __init__(self, thea, delta, ignore_label=255):
        super(DiscriminativeLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thea = thea
        self.delta = delta
        self.relu = nn.ReLU(inplace=True)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        ntarget = target.data.cpu().numpy()
        predict = predict.permute(0, 2, 3, 1)
        cls_ids = np.unique(ntarget)
        # cls_ids = cls_ids[cls_ids != 0]
        cls_ids = cls_ids[cls_ids != self.ignore_label]
        cls_ids = [cls_id for cls_id in cls_ids if np.sum(ntarget == cls_id) > 20]
        centers = {}
        loss_var = 0
        for cls_id in cls_ids:
            index = (target == cls_id)
            index = index.unsqueeze(3)
            cls_prediction = predict[index].view((-1, c))
            mean = cls_prediction.mean(0)
            centers[cls_id] = mean
            result = self.relu(torch.norm(mean - cls_prediction, 2, 1) - self.thea)
            loss_var += torch.pow(result, 2).mean()
        loss_var /= len(cls_ids)

        loss_dis = 0
        for f_cls_id in cls_ids:
            for s_cls_id in cls_ids:
                if f_cls_id != s_cls_id:
                    result = self.relu(2 * self.delta - torch.norm(centers[f_cls_id] - centers[s_cls_id]))
                    loss_dis += torch.pow(result, 2)
        loss_dis /= max((len(cls_ids) * (len(cls_ids) - 1)), 1)

        loss_reg = 0
        for cls_id in cls_ids:
            loss_reg += torch.norm(centers[cls_id])
        loss_reg /= len(cls_ids)

        return loss_var + loss_dis + 0.001 * loss_reg


class HNMDiscriminativeLoss(nn.Module):

    def __init__(self, thea, delta, weights, ignore_label=255, loss_weight=1.0):
        super(HNMDiscriminativeLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thea = thea
        self.delta = delta
        self.weights = weights
        self.relu = nn.ReLU(inplace=True)
        self.loss_weight = loss_weight

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        ntarget = target.data.cpu().numpy()
        predict = predict.permute(0, 2, 3, 1)
        cls_ids = np.unique(ntarget)
        cls_ids = cls_ids[cls_ids != self.ignore_label]
        cls_ids = [cls_id for cls_id in cls_ids if np.sum(ntarget == cls_id) > 20]
        centers = {}
        loss_var = 0
        for cls_id in cls_ids:
            index = (target == np.float(cls_id))
            index = index.unsqueeze(3)
            cls_prediction = predict[index].view((-1, c))
            mean = cls_prediction.mean(0)
            centers[cls_id] = mean
            result = self.relu(torch.norm(mean - cls_prediction, 2, 1) - self.thea)
            normliaze = max(np.sum(result.data.cpu().numpy() > 0), 1)
            loss_var += self.weights[cls_id] * torch.pow(result, 2).sum() / np.float(normliaze)
        loss_var /= len(cls_ids)

        loss_dis = 0
        for f_cls_id in cls_ids:
            for s_cls_id in cls_ids:
                if f_cls_id != s_cls_id:
                    result = self.relu(2 * self.delta - torch.norm(centers[f_cls_id] - centers[s_cls_id]))
                    loss_dis += torch.pow(result, 2)
        loss_dis /= max((len(cls_ids) * (len(cls_ids) - 1)), 1)

        loss_reg = 0
        for cls_id in cls_ids:
            loss_reg += torch.norm(centers[cls_id])
        loss_reg /= len(cls_ids)

        return self.loss_weight * (loss_var + loss_dis + 0.001 * loss_reg)


class RegionDiscriminativeLoss(nn.Module):

    def __init__(self, thea, delta, ignore_label=255):
        super(RegionDiscriminativeLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thea = thea
        self.delta = delta
        self.relu = nn.ReLU(inplace=True)

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        ntarget = target.data.cpu().numpy()
        predict = predict.permute(0, 2, 3, 1)

        loss_var_batch = 0
        loss_dis_batch = 0
        loss_reg_batch = 0
        for batch_index in range(n):
            cclabel, NRids = self.generate_region(ntarget[batch_index])

            cls_ids = np.unique(cclabel)
            # cls_ids = cls_ids[cls_ids != 0]
            cls_ids = cls_ids[cls_ids != -1]
            cls_ids = [cls_id for cls_id in cls_ids if np.sum(cclabel == cls_id) > 20]
            centers = {}

            cclabel = torch.from_numpy(cclabel).cuda()
            loss_var = 0
            for cls_id in cls_ids:
                index = (cclabel == int(cls_id))
                index = index.unsqueeze(2)
                cls_prediction = predict[batch_index][index].view((-1, c))
                mean = cls_prediction.mean(0)
                centers[cls_id] = mean
                result = self.relu(torch.norm(mean - cls_prediction, 2, 1) - self.thea)
                # result = torch.max(result, Variable(torch.FloatTensor([self.thea]).cuda(), requires_grad=False))
                loss_var += torch.pow(result, 2).mean()
                # print('cls_id: {}, loss_dis: {}'.format(cls_id, result.data.cpu().numpy().shape))
            loss_var /= len(cls_ids)
            loss_var_batch += loss_var / n

            loss_dis = 0
            connect_counts = 0
            for f_cls_id in cls_ids:
                nrids = NRids[f_cls_id]
                for s_cls_id in cls_ids:
                    if f_cls_id != s_cls_id:
                        result = self.relu(2 * self.delta - torch.norm(centers[f_cls_id] - centers[s_cls_id]))
                        loss_dis += torch.pow(result, 2)
                        connect_counts += 1
            loss_dis /= max(connect_counts, 1)
            loss_dis_batch += loss_dis / n

            loss_reg = 0
            for cls_id in cls_ids:
                loss_reg += torch.norm(centers[cls_id])
            loss_reg /= len(cls_ids)
            loss_reg_batch += loss_reg / n
        return loss_var_batch + loss_dis_batch + 0.001 * loss_reg_batch
