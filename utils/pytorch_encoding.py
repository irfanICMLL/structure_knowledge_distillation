import pdb
import threading
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn import functional as F
import os
import sys
import functools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from modules import InPlaceABN, InPlaceABNSync

BatchNorm2d = functools.partial(nn.BatchNorm2d)

def upsample(input, size=None, scale_factor=None, mode='nearest'):
    """Multi-GPU version torch.nn.functional.upsample

    Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for upsampling is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric upsampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [depth] x [height] x width`

    The modes available for upsampling are: `nearest`, `linear` (3D-only),
    `bilinear` (4D-only), `trilinear` (5D-only)

    Args:
        input (Variable): input
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
        mode (string): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear'. Default: 'nearest'
    """
    if isinstance(input, Variable):
        return F.upsample(input, size=size, scale_factor=scale_factor,
                          mode=mode)
    elif isinstance(input, tuple) or isinstance(input, list):
        lock = threading.Lock()
        results = {}
        def _worker(i, x):
            try:
                with torch.cuda.device_of(x):
                    result =  F.upsample(x, size=size, \
                        scale_factor=scale_factor,mode=mode)
                with lock:
                    results[i] = result
            except Exception as e:
                with lock:
                    resutls[i] = e 
        # multi-threading for different gpu
        threads = [threading.Thread(target=_worker,
                                    args=(i, x),
                                    )
                   for i, (x) in enumerate(input)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join() 
        outputs = dict_to_list(results)
        return outputs
    else:
        raise RuntimeError('unknown input type')


class AdaptiveAvgPool2d(Module):
    """Applies a 2D adaptive average pooling over an input signal composed of several input planes.
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single number H for a square image H x H

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = autograd.Variable(torch.randn(1, 64, 8, 9))
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = autograd.Variable(torch.randn(1, 64, 10, 9))
        >>> output = m(input)

    """
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, input):
        if isinstance(input, Variable):
            return F.adaptive_avg_pool2d(input, self.output_size)
        elif isinstance(input, tuple) or isinstance(input, list):
            return my_data_parallel(self, input)
        else:
            raise RuntimeError('unknown input type')

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'output_size=' + str(self.output_size) + ')'


class PyramidPooling(Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                BatchNorm2d(out_channels),
                                nn.ReLU(False))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                BatchNorm2d(out_channels),
                                nn.ReLU(False))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                BatchNorm2d(out_channels),
                                nn.ReLU(False))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                BatchNorm2d(out_channels),
                                nn.ReLU(False))

    def _cat_each(self, x, feat1, feat2, feat3, feat4):
        assert(len(x)==len(feat1))
        z = []
        for i in range(len(x)):
            z.append( torch.cat((x[i], feat1[i], feat2[i], feat3[i], feat4[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')
        feat1 = upsample(self.conv1(self.pool1(x)),(h,w),
                              mode='bilinear')
        feat2 = upsample(self.conv2(self.pool2(x)),(h,w),
                              mode='bilinear')
        feat3 = upsample(self.conv3(self.pool3(x)),(h,w), 
                              mode='bilinear')
        feat4 = upsample(self.conv4(self.pool4(x)),(h,w), 
                              mode='bilinear')
        if isinstance(x, Variable):
            return torch.cat((x, feat1, feat2, feat3, feat4), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            return self._cat_each(x, feat1, feat2, feat3, feat4)
        else:
            raise RuntimeError('unknown input type')

