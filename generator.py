"""SegmentationNN"""
import torch
import torch.nn as nn
#import torch.nn.functional as F

class GeneratorNET(nn.Module):

    def __init__(self, num_classes=2, in_channels=1, weight_scale = 0, dropout = 0.25, leak = 0.02):
        super(GeneratorNET, self).__init__()

        batchNorm_momentum = 0.1

        self.G = nn.Sequential(
            nn.Linear(128*128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128*128),
            nn.Tanh()

            # nn.Linear(128*128, 128*128)
            # nn.Conv2d(1, 256, kernel_size=5, padding=2),
            # nn.BatchNorm2d(256, momentum= batchNorm_momentum),
            # nn.ReLU(),
            # nn.Conv2d(256, 128, kernel_size=5, padding=2),
            # nn.BatchNorm2d(128, momentum= batchNorm_momentum),
            # nn.ReLU(),
            # nn.Conv2d(128, 32, kernel_size=5, padding=2),
            # nn.BatchNorm2d(32, momentum= batchNorm_momentum),
            # nn.ReLU(),
            # nn.Conv2d(32, 3, kernel_size=5, padding=2),
            # nn.Tanh()
            )


        if weight_scale > 0.:
            self.init_weightscale(weight_scale)


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        out = self.G(x)
        return out


    def init_weightscale(self, weight_scale):

        units = [self.G]

        for _unit in units:

            for _layer in _unit:

                if isinstance(_layer, nn.Conv2d):
                        _layer.weight.data.mul_(weight_scale)


    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


#class Maxout2d(nn.Module):
#
#    def __init__(self, in_channels, out_channels, pool_size):
#        super(Maxout2d, self).__init__()
#        self.in_channels, self.out_channels, self.pool_size = in_channels, out_channels, pool_size
#        self.conv2d = nn.Conv2d(in_channels, out_channels * pool_size, kernel_size=1, stride=1, padding=0)
#        self.bn = nn.BatchNorm2d(out_channels * pool_size)
#
#    def forward(self, x):
#        N,C,H,W  = list(x.size())
#        out = self.conv2d(x)
#        out = self.bn(out)
#
#        out = out.permute(0, 2, 3, 1)
#        m, i = out.contiguous().view(N*H*W,self.out_channels,self.pool_size).max(2)
#        m = m.squeeze(2)
#        m = m.view(N,H,W,self.out_channels)
#        m = m.permute(0, 3, 1, 2)
#        return m

class Maxout2d(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m
