"""SegmentationNN"""
import torch
import torch.nn as nn
#import torch.nn.functional as F

class NerveNET(nn.Module):

    def __init__(self, input_dim = (1, 128, 128), num_classes=2,
                 weight_scale = 0, dropout = 0.25, leak = 0.01):
        super(NerveNET, self).__init__()
        
        in_channels, W, H = input_dim

        self.down1 = nn.Sequential(nn.Dropout(p = dropout),
                                   nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(negative_slope = leak),                               
                                   nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(negative_slope = leak))                               
        self.down2 = nn.Sequential(nn.Dropout(p = dropout),
                                   nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope = leak),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.MaxPool2d(2, 2),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope = leak))
        self.down3 = nn.Sequential(nn.Dropout(p = dropout),
                                   nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope = leak),                                 
                                   nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                   nn.MaxPool2d(2, 2),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope = leak))
        self.down4 = nn.Sequential(nn.Dropout(p = dropout),
                                   nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope = leak),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                   nn.MaxPool2d(2, 2),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope = leak))
        
        self.down5 = nn.Sequential(nn.Dropout(p = dropout),
                                   nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(negative_slope = leak),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                   nn.MaxPool2d(2, 2),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(negative_slope = leak))
        
        W_d5 = int((W / 2**4 - 2) / 2 + 1)
        H_d5 = int((H / 2**4 - 2) / 2 + 1)
        
        self.out00 = nn.Sequential(nn.Linear(W_d5 * H_d5 * 512, 32),
                                   nn.LeakyReLU(negative_slope = leak),
                                   nn.BatchNorm2d(32),
                                   nn.Linear(32, 16),
                                   nn.LeakyReLU(negative_slope = leak),
                                   nn.BatchNorm2d(16),
                                   nn.Linear(16, 1),
                                   nn.Sigmoid())
        
        self.up5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.Upsample(scale_factor = 2))      

        self.out01 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1),
                                   nn.Sigmoid())
        
        self.up4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.Upsample(scale_factor = 2),
                                 nn.Conv2d(256, 128, kernel_size=3, padding=1))

        self.out02 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding=1),
                                   nn.Sigmoid())        
        
        self.up3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(128), 
                                 nn.Upsample(scale_factor = 2),
                                 nn.Conv2d(128, 64, kernel_size=3, padding=1))
        self.up2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.Upsample(scale_factor = 2),
                                 nn.Conv2d(64, 32, kernel_size=3, padding=1))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor = 2))
        
        self.out = nn.Sequential(nn.Dropout(p = dropout),
                                 nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                 nn.LeakyReLU(negative_slope = leak),
                                 nn.BatchNorm2d(16),
                                 nn.Conv2d(16, 16, kernel_size=3, padding=1),
                                 nn.LeakyReLU(negative_slope = leak),
                                 nn.BatchNorm2d(16),
                                 nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
                                 nn.Sigmoid())
                                 
        if weight_scale > 0.:
            self.init_weightscale(weight_scale)


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4) #B
        
        
        up5 = self.up5(down5) #256
        up5_down4 = torch.cat([up5, down4], dim = 1) #512
        up4 = self.up4(up5_down4) #256--128
        up4_down3 = torch.cat([up4, down3], dim = 1) #256
        up3 = self.up3(up4_down3) #256--64
        up3_down2 = torch.cat([up3, down2], dim = 1) #128       
        up2 = self.up2(up3_down2) #128--32
        
        #up2_down1 = torch.cat([up2, down1], dim = 1)        
        #up1 = self.up1(up2_down1)
        
        up1 = self.up1(up2)
        
        out00 = self.out00(down5.view(down5.size()[0], -1))
        out01 = self.out01(up5)
        out02 = self.out02(up4)
        out = self.out(up1)
        
        return out, out00, out01, out02

    
    def init_weightscale(self, weight_scale):
        
        units = [self.down1,
                  self.down2,
                  self.down3,
                  self.down4,
                  self.down5,
                  self.out00,
                  self.out01,
                  self.out02,
                  self.up5,
                  self.up4,
                  self.up3,
                  self.up2,
                  #self.up1,
                  self.out]
        
        for _unit in units:
            
            for _layer in _unit:
                
                if isinstance(_layer, nn.Conv2d or nn.Linear):
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

    def __init__(self, input_channels, output_channels, pool_size):
        '''
        Parameters:
        -----------
        input_channels = number of input channels
        output_channels = number of output channels
        pool_size = number of affine layers to calculate and take the max from
        '''
        super().__init__()
        self.input_channels, self.output_channels, self.pool_size = input_channels, output_channels, pool_size
        self.lin = nn.Linear(input_channels, output_channels * pool_size)


    def forward(self, inputs):
        
        shape = list(inputs.size()) #get inputs shape e.g. [batch_size, num_channels, H, W]
        if len(shape) < 4:
            inputs = inputs.view(-1, *shape) #reshape input to have 4 dimensions
        inputs = inputs.permute(0, 2, 3, 1) #permute input to batch_size, H, W, num_channels]
        
        #create output shape
        newshape = list(inputs.size()) #get new inputs shape
        newshape[-1] = self.output_channels  #substitute input_channels with output_channels
        newshape.append(self.pool_size) #add size of dimension to max over
        max_dim = len(newshape) - 1 #max over last dimension
        
        out = self.lin(inputs) #forward to nn.Linear
        out = out.view(*newshape) #reshape the output
        out, i = out.max(max_dim) #take the max out of the pool
        out = out.permute(0, 3, 1, 2) #permute back to [batch_size, num_channels, H, W]
        return out


