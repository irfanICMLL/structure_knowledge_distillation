import argparse
import torch
import time
import logging
import os
from utils.utils import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description='knowledge-distillation')
        parser.add_argument('--data_set', default='cityscape',type=str, metavar='', help='')

        parser.add_argument('--classes_num', default=19, type=int,metavar='N', help='class num of the dataset')
        parser.add_argument('--T_ckpt_path', default='./ckpt/Teacher/CS_scenes_38413_0.7832174615268139.pth',type=str, metavar='teacher ckpt path', help='teacher ckpt path')
        parser.add_argument('--S_resume', default='True', type=str2bool, metavar='is or not use student', help='is or not use student ckpt')
        parser.add_argument('--S_ckpt_path', default='./dataset/resnet18-imagenet.pth',type=str, metavar='student ckpt path', help='student ckpt path')
        parser.add_argument('--D_resume', default=True, type=bool,metavar='is or not use discriminator', help='is or not use discriminator ckpt')
        parser.add_argument('--D_ckpt_path', default='',type=str, metavar='discriminator ckpt path', help='discriminator ckpt path')
        parser.add_argument("--batch-size", type=int, default=8, help="Number of images sent to the network in one step.")
        parser.add_argument('--start_epoch', default=0, type=int,metavar='start_epoch', help='start_epoch')
        parser.add_argument('--epoch_nums', default=1, type=int,metavar='epoch_nums', help='epoch_nums')
        parser.add_argument('--parallel', default='True', type=str, metavar='parallel', help='attribute of saved name')
        parser.add_argument("--data-dir", type=str, default='', help="Path to the directory containing the PASCAL VOC dataset.")
        parser.add_argument("--data-list", type=str, default='./dataset/list/cityscapes/train.lst', help="Path to the file listing the images in the dataset.")
        parser.add_argument("--ignore-label", type=int, default=255, help="The index of the label to ignore during the training.")
        parser.add_argument("--input-size", type=str, default='512,512', help="Comma-separated string with height and width of images.")
        parser.add_argument("--is-training", action="store_true", help="Whether to updates the running means and variances during the training.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--num-steps", type=int, default=40000, help="Number of training steps.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
        parser.add_argument("--random-mirror", action="store_true", help="Whether to randomly mirror the inputs during the training.")
        parser.add_argument("--random-scale", action="store_true", help="Whether to randomly scale the inputs during the training.")
        parser.add_argument("--snapshot-dir", type=str, default='./snapshots/', help="Where to save snapshots of the model.")
        parser.add_argument("--weight-decay", type=float, default=1.e-4, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--gpu", type=str, default='None', help="choose gpu device.")
        parser.add_argument("--recurrence", type=int, default=1, help="choose the number of recurrence.")

        parser.add_argument("--last-step", type=int, default=0, help="last train step.")
        parser.add_argument("--is-student-load-imgnet", type=str2bool, default='True', help="is student load imgnet")
        parser.add_argument("--student-pretrain-model-imgnet", type=str, default='None', help="student pretrain model on imgnet")
        parser.add_argument("--pi", type=str2bool, default='True', help="is pixel wise loss using or not")
        parser.add_argument("--pa", type=str2bool, default='True', help="is pixel wise loss using or not")
        parser.add_argument("--ho", type=str2bool, default='True', help="is pixel wise loss using or not")
        parser.add_argument("--adv-loss-type", type=str, default='wgan-gp', help="adversarial loss setting")
        parser.add_argument("--imsize-for-adv", type=int, default=65, help="imsize for addv")
        parser.add_argument("--adv-conv-dim", type=int, default=64, help="conv dim in adv")
        parser.add_argument("--lambda-gp", type=float, default=10.0, help="lambda_gp")
        parser.add_argument("--lambda-d", type=float, default=0.1, help="lambda_d")
        parser.add_argument("--lambda-pi", type=float, default=10.0, help="lambda_pi")
        parser.add_argument('--lambda-pa', default=1.0, type=float, help='')
        parser.add_argument('--pool-scale', default=0.5, type=float, help='')
        parser.add_argument("--preprocess-GAN-mode", type=int, default=1, help="preprocess-GAN-mode should be tanh or bn")
        parser.add_argument("--lr-g", type=float, default=1e-2, help="learning rate for G")
        parser.add_argument("--lr-d", type=float, default=4e-4, help="learning rate for D")
        parser.add_argument("--best-mean-IU", type=float, default=0.0, help="learning rate for D")

        args = parser.parse_args()

        args.save_name = 'save_path'
        args.S_ckpt_path = './ckpt/'+ args.save_name +'/Student'
        args.D_ckpt_path = './ckpt/' + args.save_name +'/Distriminator'
        args.D_att_ckpt_path  = './ckpt/' + args.save_name +'/Att_discriminator'
        args.log_path = './ckpt/log/' + args.save_name

        log_init(args.log_path, args.data_set)
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.gpu_num = len(args.gpu.split(','))
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        logger_path = args.log_path + '/tensorboard/'

        for key, val in args._get_kwargs():
            logging.info(key+' : '+str(val))

        return args


class TrainOptionsForTest():
    def initialize(self):
        parser = argparse.ArgumentParser(description='knowledge-distillation')
        parser.add_argument("--data-dir", type=str, default='', help="")
        parser.add_argument("--resume-from", type=str, default='', help="")
        args = parser.parse_args()
        for key, val in args._get_kwargs():
            print(key+' : '+str(val))
        return args
