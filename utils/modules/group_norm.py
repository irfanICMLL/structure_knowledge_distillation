import torch
import torch.nn as nn
import tensorflow as tf
from torch.autograd import Variable


def tf_group_norm(x, gamma, beta, groups, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    batch_size, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    x = tf.reshape(x, [batch_size, groups, channels_per_group, height, width])
    mean, var = tf.nn.moments(x, axes=[2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [batch_size, num_channels, height, width])
    return x * gamma + beta


class GroupNorm2D(nn.Module):
    """
        Group Normalization
        Reference: https://128.84.21.199/abs/1803.08494v1
    """
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        """

        :param num_features:
        :param num_groups:
        :param eps:
        """
        super(GroupNorm2D, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        # x: input features with shape [N, C, H, W]
        # weight, bias: scale and offset, with shape [1, C, 1, 1]
        # groups: number of groups for GroupNorm

        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.num_groups == 0

        x = x.view(batch_size, self.num_groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(batch_size, num_channels, height, width)
        return x * self.weight + self.bias


if __name__ == "__main__":
    dummy_in = Variable(torch.randn(1, 32, 448, 896).cuda(), requires_grad=True)

    group_norm = GroupNorm2D(num_features=32, num_groups=8, eps=1e-5)
    dummy_out = group_norm(dummy_in)

    dummy_in = tf.random_normal([1, 32, 448, 896], mean=-1, stddev=4)
    gamma = tf.random_normal([1, 32, 1, 1], mean=-1, stddev=4)
    beta = tf.random_normal([1, 32, 1, 1], mean=-1, stddev=4)

    dummy_out = tf_group_norm(dummy_in, gamma, beta, 8, eps=1e-5)