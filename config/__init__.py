import os
import argparse
import torch

DATASET = "cityscapes_light"
DATA_DIRECTORY = '/fast/users/a1760953/AIEdgeContest/cityscapes/'
BATCH_SIZE = 8
DATA_LIST_PATH = '/fast/users/a1760953/AIEdgeContest/cityscapes/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 40000
POWER = 0.9
RANDOM_SEED = 304
RESTORE_FROM = "/fast/users/a1760953/cvpr19/pretrained/resnet101-imagenet.pth"
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots_psp_ohem_trainval/'
WEIGHT_DECAY = 0.0005
PRE_TRAINED = "/fast/users/a1760953/cvpr19/MobileNet-V2-Pytorch/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar"
TEACHER_FROM = "/fast/users/a1760953/cvpr19/checkpoint/snapshots_resnet_psp_dsn_1e-4_5e-4_8_20000_DSN_0.4_769light/CS_scenes_20000.pth"
DIS_FROM = "/fast/users/a1760953/cvpr19/Res18SD/checkpoint/snapshots_student_res18_pre_1e-5_400_GAN_label_res18_pre/CS_Dnet_10000.pth"


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Parameters():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Pytorch Segmentation Network")
        parser.add_argument("--dataset", type=str, default=DATASET,
                            help="Specify the dataset to use.")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                            help="Number of images sent to the network in one step.")
        parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                            help="Path to the directory containing the PASCAL VOC dataset.")
        parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                            help="Path to the file listing the images in the dataset.")
        parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                            help="The index of the label to ignore during the training.")
        parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                            help="Comma-separated string with height and width of images.")
        parser.add_argument("--is-training", action="store_true",
                            help="Whether to updates the running means and variances during the training.")
        parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--momentum", type=float, default=MOMENTUM,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--not-restore-last", action="store_true",
                            help="Whether to not restore last (FC) layers.")
        parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                            help="Number of classes to predict (including background).")
        parser.add_argument("--start-iters", type=int, default=0,
                            help="Number of classes to predict (including background).")
        parser.add_argument("--epoch", type=int, default=80,
                            help="Number of training steps.")
        parser.add_argument("--save_epoch", type=int, default=10,
                            help="Number of training steps.")
        parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                            help="Number of training steps.")
        parser.add_argument("--power", type=float, default=POWER,
                            help="Decay parameter to compute the learning rate.")
        parser.add_argument("--random-mirror", action="store_true",
                            help="Whether to randomly mirror the inputs during the training.")
        parser.add_argument("--random-scale", action="store_true",
                            help="Whether to randomly scale the inputs during the training.")
        parser.add_argument("--fix", action="store_true",
                            help="Fix bn")
        parser.add_argument("--side", action="store_true",
                            help="use the side loss")
        parser.add_argument("--weight", type=float, default=0,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                            help="Random seed to have reproducible results.")
        parser.add_argument("--restore-from", type=str, default=None,
                            help="Where restore model parameters from.")
        parser.add_argument("--pre_trained", type=str, default=PRE_TRAINED,
                            help="Where pre trained model parameters from.")
        parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                            help="How many images to save.")
        parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                            help="Save summaries and checkpoint every often.")
        parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                            help="Where to save snapshots of the model.")
        parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                            help="Regularisation parameter for L2-loss.")
        parser.add_argument("--gpu", type=str, default='0',
                            help="choose gpu device.")

        parser.add_argument("--ohem-thres", type=float, default=0.6,
                            help="choose the samples with correct probability underthe threshold.")
        parser.add_argument("--ohem-thres1", type=float, default=0.8,
                            help="choose the threshold for easy samples.")
        parser.add_argument("--ohem-thres2", type=float, default=0.5,
                            help="choose the threshold for hard samples.")
        parser.add_argument("--ohem-keep", type=int, default=100000,
                            help="choose the samples with correct probability underthe threshold.")
        parser.add_argument("--use-weight", type=str2bool, nargs='?', const=True,
                            help="whether use the weights to solve the unbalance problem between classes.")
        parser.add_argument("--use-val", type=str2bool, nargs='?', const=True,
                            help="choose whether to use the validation set to train.")
        parser.add_argument("--use-extra", type=str2bool, nargs='?', const=True,
                            help="choose whether to use the extra set to train.")
        parser.add_argument("--ohem", type=str2bool, nargs='?', const=True,
                            help="choose whether conduct ohem.")
        parser.add_argument("--network", type=str, default='resnet',
                            help="choose which network to use.")
        parser.add_argument("--method", type=str, default='psp_dsn_floor',
                            help="choose method to train.")
        parser.add_argument("--lovasz-loss", action="store_true",
                            help="Whether to use the lovasz loss.")
        parser.add_argument("--cross-entropy-lovasz-loss", action="store_true",
                            help="Whether to use both the lovasz loss and cross entropy loss.")
        parser.add_argument("--use-psp", action="store_true",
                            help="Whether to use psp layer when use the correlation operation.")
        parser.add_argument("--reduce", action="store_false",
                            help="Whether to use reduce when computing the cross entropy loss.")
        parser.add_argument("--ohem-single", action="store_true",
                            help="Whether to use hard sample mining only for the last supervision.")
        parser.add_argument("--hard-image", action="store_true",
                            help="Whether to do hard images mining.")
        parser.add_argument("--use-parallel", action="store_true",
                            help="Whether to the default parallel.")

        parser.add_argument("--use-dis-loss", action="store_true",
                            help="Whether to use the dicriminative loss.")
        parser.add_argument("--alpha", type=str, default='1',
                            help="choose the weight of the discriminative loss.")

        parser.add_argument("--use-image-iou", action="store_true",
                            help="Whether to use image iou to reweight the images.")

        parser.add_argument("--dsn-weight", type=float, default=0.4,
                            help="choose the weight of the dsn supervision.")
        parser.add_argument('--seed', default=304, type=int, help='manual seed')

        # extra parameters for evaluation settings

        parser.add_argument("--output-path", type=str, default='./seg_output_eval_set',
                            help="Path to the segmentation map prediction.")
        parser.add_argument("--store-output", type=str, default='False',
                            help="whether store the predicted segmentation map.")
        parser.add_argument("--use-flip", type=str, default='False',
                            help="whether use test-stage flip.")
        parser.add_argument("--use-ms", type=str, default='False',
                            help="whether use test-stage multi-scale crop.")
        parser.add_argument("--predict-choice", type=str, default='whole',
                            help="crop: choose the training crop size; whole: choose the whole picture; step: choose to predict the images with multiple steps.")
        parser.add_argument("--whole-scale", type=str, default='1',
                            help="choose the scale to rescale whole picture.")

        # parameters for ade20k dataset
        parser.add_argument("--start-epochs", type=int, default=0,
                            help="Number of the initial staring epochs.")
        parser.add_argument("--end-epochs", type=int, default=120,
                            help="Number of the overall training epochs.")
        parser.add_argument("--save-epoch", type=int, default=20,
                            help="Save summaries and checkpoint every often.")
        parser.add_argument("--criterion", type=str, default='ce',
                            help="Specify the specific criterion/loss functions to use.")
        parser.add_argument('--eval', action='store_true', default=False,
                            help='evaluating mIoU')

        # l2 regularization mode
        parser.add_argument('--decay-mode', type=str, default="11",  # we can choose mode 11 or mode 1
                            help='choose l2 weight decay or l2-sp weight decay')

        # extra data augmentation methods to use
        parser.add_argument("--use-extra-aug", action="store_true",
                            help="Whether to use extra data augmentation.")

        # sn normalization
        parser.add_argument("--use-sn", action="store_true",
                            help="Whether to use switchable normalization.")

        # extra parameters for center loss
        parser.add_argument("--center_weight", type=float, default=0.001,
                            help="choose the weight of the center loss supervision.")

        # extra parameters to choose whether use the finetune mode(fix mean and var within BN)
        parser.add_argument("--use-finetune", action="store_true",
                            help="choose whether to finetune the model.")

        parser.add_argument("--fix-lr", action="store_true",
                            help="choose whether to fix the learning rate.")

        parser.add_argument("--no-class-balance", action="store_false",
                            help="choose whether to use the class balance weights.")

        # parameters for global psp, local self-attention context mixture.
        parser.add_argument("--sa-weight", type=float, default=1,
                            help="choose the weight of the sa-weight supervision.")
        parser.add_argument("--psp-weight", type=float, default=1,
                            help="choose the weight of the psp-weight supervision.")
        parser.add_argument("--fuse-weight", type=float, default=1,
                            help="choose the weight of the fuse-weight supervision.")
        # parameters for structured distillation
        parser.add_argument("--sd_mode", type=str, default='light_label',
                            help="choose method to train.")
        parser.add_argument("--adv_loss", type=str, default='hinge',
                            help="choose the GAN loss")
        parser.add_argument("--teacher-from", type=str, default=TEACHER_FROM,
                            help="Path to the model of teacher net")
        parser.add_argument("--dis-from", type=str, default=DIS_FROM,
                            help="Path to the model of discriminator net")
        parser.add_argument("--weight_l", type=float, default=1,
                            help="Weight for label loss.")
        parser.add_argument("--weight_k", type=float, default=1,
                            help="Weight for knowledge distillation loss.")
        parser.add_argument("--weight_s", type=float, default=1,
                            help="Weight for structured distillation loss.")
        parser.add_argument("--weight_kf", type=float, default=1,
                            help="Weight for feature mimic.")
        parser.add_argument("--weight_GAN", type=float, default=1,
                            help="Weight for GAN loss.")
        parser.add_argument("--tem", type=float, default=1,
                            help="temperture for KD.")
        parser.add_argument("--location", type=float, default=1,
                            help="0=score map,1=feature mao")
        parser.add_argument("--divided_lr", type=int, default=50,
                            help="Number of the initial staring epochs.")
        parser.add_argument("--teacher_model", type=str, default='psp_dsn_floor',
                            help="Number of the initial staring epochs.")
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args
