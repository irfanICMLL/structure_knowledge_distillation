import argparse
import logging
import os
import pdb
from torch.autograd import Variable
import os.path as osp
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import resource
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import *
import torch.backends.cudnn as cudnn

from utils.criterion import CriterionDSN, CriterionOhemDSN, CriterionPixelWise, \
    CriterionAdv, CriterionAdvForG, CriterionAdditionalGP, CriterionPairWiseforWholeFeatAfterPool
import utils.parallel as parallel_old
from networks.pspnet_combine import Res_pspnet, BasicBlock, Bottleneck
from networks.sagan_models import Discriminator
from networks.evaluate import evaluate_main

torch_ver = torch.__version__[:3]

class NetModel():
    def name(self):
        return 'kd_seg'

    def DataParallelModelProcess(self, model, ParallelModelType = 1, is_eval = 'train', device = 'cuda'):
        if ParallelModelType == 1:
            parallel_model = DataParallelModel(model)
        elif ParallelModelType == 2:
            parallel_model = parallel_old.DataParallelModel(model)
        else:
            raise ValueError('ParallelModelType should be 1 or 2')
        if is_eval == 'eval':
            parallel_model.eval()
        elif is_eval == 'train':
            parallel_model.train()
        else:
            raise ValueError('is_eval should be eval or train')
        parallel_model.float()
        parallel_model.to(device)
        return parallel_model

    def DataParallelCriterionProcess(self, criterion, device = 'cuda'):
        criterion = parallel_old.my_DataParallelCriterion(criterion)
        criterion.cuda()
        return criterion

    def __init__(self, args):
        cudnn.enabled = True
        self.args = args
        device = args.device
        student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = args.classes_num)
        load_S_model(args, student, False)
        print_model_parm_nums(student, 'student_model')
        self.parallel_student = self.DataParallelModelProcess(student, 2, 'train', device)
        self.student = student

        teacher = Res_pspnet(Bottleneck, [3, 4, 23, 3], num_classes = args.classes_num)
        load_T_model(teacher, args.T_ckpt_path)
        print_model_parm_nums(teacher, 'teacher_model')
        self.parallel_teacher = self.DataParallelModelProcess(teacher, 2, 'eval', device)
        self.teacher = teacher

        D_model = Discriminator(args.preprocess_GAN_mode, args.classes_num, args.batch_size, args.imsize_for_adv, args.adv_conv_dim)
        load_D_model(args, D_model, False)
        print_model_parm_nums(D_model, 'D_model')
        self.parallel_D = self.DataParallelModelProcess(D_model, 2, 'train', device)

        self.G_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, self.student.parameters()), 'initial_lr': args.lr_g}], args.lr_g, momentum=args.momentum, weight_decay=args.weight_decay)
        self.D_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, D_model.parameters()), 'initial_lr': args.lr_d}], args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay)

        self.best_mean_IU = args.best_mean_IU

        self.criterion = self.DataParallelCriterionProcess(CriterionDSN()) #CriterionCrossEntropy()
        self.criterion_pixel_wise = self.DataParallelCriterionProcess(CriterionPixelWise())
        #self.criterion_pair_wise_for_interfeat = [self.DataParallelCriterionProcess(CriterionPairWiseforWholeFeatAfterPool(scale=args.pool_scale[ind], feat_ind=-(ind+1))) for ind in range(len(args.lambda_pa))]
        self.criterion_pair_wise_for_interfeat = self.DataParallelCriterionProcess(CriterionPairWiseforWholeFeatAfterPool(scale=args.pool_scale, feat_ind=-5))
        self.criterion_adv = self.DataParallelCriterionProcess(CriterionAdv(args.adv_loss_type))
        if args.adv_loss_type == 'wgan-gp':
            self.criterion_AdditionalGP = self.DataParallelCriterionProcess(CriterionAdditionalGP(self.parallel_D, args.lambda_gp))
        self.criterion_adv_for_G = self.DataParallelCriterionProcess(CriterionAdvForG(args.adv_loss_type))
            
        self.mc_G_loss = 0.0
        self.pi_G_loss = 0.0
        self.pa_G_loss = 0.0
        self.D_loss = 0.0

        cudnn.benchmark = True
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

    def AverageMeter_init(self):
        self.parallel_top1_train = AverageMeter()
        self.top1_train = AverageMeter()

    def set_input(self, data):
        args = self.args
        images, labels, _, _ = data
        self.images = images.cuda()
        self.labels = labels.long().cuda()
        if torch_ver == "0.3":
            self.images = Variable(images)
            self.labels = Variable(labels)

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr*((1-float(iter)/max_iter)**(power))
            
    def adjust_learning_rate(self, base_lr, optimizer, i_iter):
        args = self.args
        lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def forward(self):
        args = self.args
        with torch.no_grad():
            self.preds_T = self.parallel_teacher.eval()(self.images, parallel=args.parallel)
        self.preds_S = self.parallel_student.train()(self.images, parallel=args.parallel)

    def student_backward(self):
        args = self.args
        G_loss = 0.0
        temp = self.criterion(self.preds_S, self.labels, is_target_scattered = False)
        temp_T = self.criterion(self.preds_T, self.labels, is_target_scattered = False)
        self.mc_G_loss = temp.item()
        G_loss = G_loss + temp
        if args.pi == True:
            temp = args.lambda_pi*self.criterion_pixel_wise(self.preds_S, self.preds_T, is_target_scattered = True)
            self.pi_G_loss = temp.item()
            G_loss = G_loss + temp
        if args.pa == True:
            #for ind in range(len(args.lambda_pa)):
            #    if args.lambda_pa[ind] != 0.0:
            #        temp1 = self.criterion_pair_wise_for_interfeat[ind](self.preds_S, self.preds_T, is_target_scattered = True)
            #        self.pa_G_loss[ind] = temp1.item()
            #        G_loss = G_loss + args.lambda_pa[ind]*temp1
            #    elif args.lambda_pa[ind] == 0.0:
            #        self.pa_G_loss[ind] = 0.0
            temp1 = self.criterion_pair_wise_for_interfeat(self.preds_S, self.preds_T, is_target_scattered = True)
            self.pa_G_loss = temp1.item()
            G_loss = G_loss + args.lambda_pa*temp1
        if self.args.ho == True:
            d_out_S = self.parallel_D(eval(compile(to_tuple_str('self.preds_S', args.gpu_num, '[0]'), '<string>', 'eval')), parallel=args.parallel)
            G_loss = G_loss + args.lambda_d*self.criterion_adv_for_G(d_out_S, d_out_S, is_target_scattered = True)
        G_loss.backward()
        self.G_loss = G_loss.item()

    def discriminator_backward(self):
        self.D_solver.zero_grad()
        args = self.args
        d_out_T = self.parallel_D(eval(compile(to_tuple_str('self.preds_T', args.gpu_num, '[0].detach()'), '<string>', 'eval')), parallel=True)
        d_out_S = self.parallel_D(eval(compile(to_tuple_str('self.preds_S', args.gpu_num, '[0].detach()'), '<string>', 'eval')), parallel=True)
        d_loss = args.lambda_d*self.criterion_adv(d_out_S, d_out_T, is_target_scattered = True)

        if args.adv_loss_type == 'wgan-gp':
            d_loss += args.lambda_d*self.criterion_AdditionalGP(self.preds_S, self.preds_T, is_target_scattered = True)

        d_loss.backward()
        self.D_loss = d_loss.item()
        self.D_solver.step()

    def optimize_parameters(self):
        self.forward()
        self.G_solver.zero_grad()
        self.student_backward()
        self.G_solver.step()
        if self.args.ho == True:
            self.discriminator_backward()

    def evalute_model(self, model, loader, gpu_id, input_size, num_classes, whole):
        mean_IU, IU_array = evaluate_main(model=model, loader = loader,  
                gpu_id = gpu_id, 
                input_size = input_size, 
                num_classes = num_classes,
                whole = whole)
        return mean_IU, IU_array 

    def print_info(self, epoch, step):
        logging.info('step:{:5d} G_lr:{:.6f} G_loss:{:.5f}(mc:{:.5f} pixelwise:{:.5f} pairwise:{:.5f}) D_lr:{:.6f} D_loss:{:.5f}'.format(
                        step, self.G_solver.param_groups[-1]['lr'], 
                        self.G_loss, self.mc_G_loss, self.pi_G_loss, self.pa_G_loss, 
                        self.D_solver.param_groups[-1]['lr'], self.D_loss))

    def __del__(self):
        pass

    def save_ckpt(self, epoch, step, mean_IU, IU_array):
        torch.save(self.student.state_dict(),osp.join(self.args.snapshot_dir, 'CS_scenes_'+str(step)+'_'+str(mean_IU)+'.pth'))  



