from torch.utils import data
from networks.pspnet_combine import Res_pspnet, BasicBlock, Bottleneck
from networks.evaluate import evaluate_main
from dataset.datasets import CSDataTestSet
from utils.train_options import TrainOptionsForTest
import torch

if __name__ == '__main__':
    args = TrainOptionsForTest().initialize()
    testloader = data.DataLoader(CSDataTestSet(args.data_dir, './dataset/list/cityscapes/test.lst', crop_size=(1024, 2048)), 
                                    batch_size=1, shuffle=False, pin_memory=True)
    student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = 19)
    student.load_state_dict(torch.load(args.resume_from))
    evaluate_main(student, testloader, '0', '512,512', 19, True, type = 'test')
