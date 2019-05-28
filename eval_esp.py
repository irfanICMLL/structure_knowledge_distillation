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
import matplotlib
import torchvision

matplotlib.use('Agg')
import argparse
import scipy
from scipy import ndimage
import torch, cv2
import numpy as np
import numpy.ma as ma
import sys
import pdb
import torch

from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from dataset import get_segmentation_dataset
from network import get_segmentation_model
from config import Parameters
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage
from utils.parallel import ModelDataParallel, CriterionDataParallel

import matplotlib.pyplot as plt
import torch.nn as nn

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    palette = [0] * (num_cls * 3)
    palette[0:3] = (128, 64, 128)  # 0: 'road'
    palette[3:6] = (244, 35, 232)  # 1 'sidewalk'
    palette[6:9] = (70, 70, 70)  # 2''building'
    palette[9:12] = (102, 102, 156)  # 3 wall
    palette[12:15] = (190, 153, 153)  # 4 fence
    palette[15:18] = (153, 153, 153)  # 5 pole
    palette[18:21] = (250, 170, 30)  # 6 'traffic light'
    palette[21:24] = (220, 220, 0)  # 7 'traffic sign'
    palette[24:27] = (107, 142, 35)  # 8 'vegetation'
    palette[27:30] = (152, 251, 152)  # 9 'terrain'
    palette[30:33] = (70, 130, 180)  # 10 sky
    palette[33:36] = (220, 20, 60)  # 11 person
    palette[36:39] = (255, 0, 0)  # 12 rider
    palette[39:42] = (0, 0, 142)  # 13 car
    palette[42:45] = (0, 0, 70)  # 14 truck
    palette[45:48] = (0, 60, 100)  # 15 bus
    palette[48:51] = (0, 80, 100)  # 16 train
    palette[51:54] = (0, 0, 230)  # 17 'motorcycle'
    palette[54:57] = (119, 11, 32)  # 18 'bicycle'
    palette[57:60] = (105, 105, 105)
    return palette


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = np.pad(img, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img


def predict_esp(net, image):
    # N_, C_, H_, W_ = image.shape
    # # interp =
    # height, width = image.shape[2:]
    # size = (int(width * 0.5), int(height * 0.5))
    # image=image[0].transpose(1,2,0)
    # image = cv2.resize(image,size, interpolation=cv2.INTER_LINEAR)
    # # interp = nn.Upsample(size=size, mode='bilinear')
    # image=image.transpose(2,0,1)
    # image = image.reshape(1, 3, 512, 1024)

    # image = nn.functional.interpolate(image,size=(512,1024), mode='bilinear', align_corners=True)

    with torch.no_grad():
        full_prediction = net(Variable(image).cuda(), )
    full_prediction = nn.functional.interpolate(full_prediction[0], size=(360, 480), mode='bilinear',
                                                align_corners=True)
    result = full_prediction.cpu().data.numpy().transpose(0, 2, 3, 1)
    # result = full_prediction[0].cpu().data.numpy().transpose(0, 2, 3, 1)
    del full_prediction
    del image

    return result


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


def id2trainId(label, id_to_trainid, reverse=False):
    label_copy = label.copy()
    if reverse:
        for v, k in id_to_trainid.items():
            label_copy[label == k] = v
    else:
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v
    return label_copy


def main():
    """Create the model and start the evaluation process."""
    args = Parameters().parse()
    # #
    # args.method = 'student_res18_pre'
    args.method = 'student_esp_d'
    args.dataset = 'camvid_light'
    args.data_list = "/ssd/yifan/SegNet/CamVid/test.txt"
    args.data_dir = "/ssd/yifan/"
    args.num_classes = 11
    # args.method='psp_dsn_floor'
    args.restore_from = "./checkpoint/Camvid/ESP/base_57.8.pth"
    # args.restore_from="/teamscratch/msravcshare/v-yifan/ESPNet/train/0.4results_enc_01_enc_2_8/model_298.pth"
    # args.restore_from = "/teamscratch/msravcshare/v-yifacd n/sd_pytorch0.5/checkpoint/snapshots_psp_dsn_floor_1e-2_40000_TEACHER864/CS_scenes_40000.pth"
    # args.restore_from = "/teamscratch/msravcshare/v-yifan/sd_pytorch0.5/checkpoint/snapshots_psp_dsn_floor_1e-2_40000_TEACHER5121024_esp/CS_scenes_40000.pth"
    # args.data_list = '/teamscratch/msravcshare/v-yifan/deeplab_v3/dataset/list/cityscapes/train.lst'
    args.batch_size = 1
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    print(args)
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # args.method='psp_dsn'
    deeplab = get_segmentation_model(args.method, num_classes=args.num_classes)

    ignore_label = 255
    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                     3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                     7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                     14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                     18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                     28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # args.restore_from="/teamscratch/msravcshare/v-yifan/sd_pytorch0.3/checkpoint/snapshots_resnet_psp_dsn_1e-4_5e-4_8_20000_DSN_0.4_769light/CS_scenes_20000.pth"
    # if 'dense' in args.method:
    #
    if args.restore_from is not None:
        saved_state_dict = torch.load(args.restore_from)
        c_keys = saved_state_dict.keys()
        for i in c_keys:
            flag = i.split('.')[0]
        if 'module' in flag:
            deeplab = nn.DataParallel(deeplab)
        deeplab.load_state_dict(saved_state_dict)
        if 'module' not in flag:
            deeplab = nn.DataParallel(deeplab)
    # if 'dense' not in args.method:
    #     deeplab = nn.DataParallel(deeplab)
    model = deeplab
    model.eval()
    model.cuda()
    # args.dataset='cityscapes_light'
    testloader = data.DataLoader(get_segmentation_dataset(args.dataset, root=args.data_dir, list_path=args.data_list,
                                                          crop_size=(360, 480), mean=IMG_MEAN, scale=False,
                                                          mirror=False),
                                 batch_size=args.batch_size, shuffle=False, pin_memory=True)

    data_list = []
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))

    palette = get_palette(20)

    image_id = 0
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % (index))
        if args.side:
            image, label, _, size, name = batch
        elif 'sd' in args.dataset:
            _, image, label, size, name = batch
        else:
            image, label, size, name = batch
        # print('image name: {}'.format(name))
        size = size[0].numpy()
        output = predict_esp(model, image)
        # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        result = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
        # result=cv2.resize(result, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        m_seg_pred = ma.masked_array(result, mask=torch.eq(label, 255))
        ma.set_fill_value(m_seg_pred, 20)
        seg_pred = m_seg_pred

        for i in range(image.size(0)):
            image_id += 1
            print('%d th segmentation map generated ...' % (image_id))
            args.store_output = 'True'
            output_path = './esp_camvid_base/'
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            if args.store_output == 'True':
                # print('a')
                output_im = PILImage.fromarray(seg_pred[i])
                output_im.putpalette(palette)
                output_im.save(output_path + '/' + name[i] + '.png')

        seg_gt = np.asarray(label.numpy()[:, :size[0], :size[1]], dtype=np.int)
        ignore_index = seg_gt != 255
        seg_gt = seg_gt[ignore_index]
        seg_pred = seg_pred[ignore_index]
        confusion_matrix += get_confusion_matrix(seg_gt, seg_pred, args.num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    print({'meanIU': mean_IU, 'IU_array': IU_array})

    print("confusion matrix\n")
    print(confusion_matrix)


if __name__ == '__main__':
    main()
