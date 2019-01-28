import torch
import torch.nn as nn
from torch.nn import init

class Discriminator(nn.Module):

    def __init__(self, input_dim = (1, 128, 128), num_channels = 2, 
                 weight_scale = True, dropout = 0.05):
        
        super().__init__()
        
        C, H, W = input_dim
        self.dropout = dropout
        
        self.down1 = nn.Sequential(nn.Dropout(p = self.dropout),
                                   nn.Conv2d(num_channels, 32, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
                               
        self.down2 = nn.Sequential(nn.Dropout(p = self.dropout),
                                   nn.Conv2d(32, 64, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        
        
        W_b = int((W / 2**1 - 2) / 2 + 1) #exponent = #poolings - 1
        H_b = int((H / 2**1 - 2) / 2 + 1)
            
        self.lin1 = nn.Sequential(nn.Linear(W_b * H_b * 64, 32),
                                 nn.ReLU())
        self.lin2= nn.Sequential(nn.Linear(32, 1),
                                 nn.Sigmoid())
        
        self.pool = nn.MaxPool2d(2, 2)
                
        if weight_scale:
            self.init_params()


    def forward(self, inputs):
        out = self.down1(inputs)
        out = self.down2(self.pool(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.lin1(out)
        out = self.lin2(out)
        return out
    
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def init_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
            
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

            
class Generator(nn.Module):

    def __init__(self, input_dim = (1, 128, 128), num_classes=2,
                 weight_scale = True,  dropout = 0.05):
        super().__init__()
        
        self.input_dim = input_dim
        C, H, W = input_dim
        
        self.dropout = dropout
        
        self.G_lin  = nn.Sequential(nn.Linear(C*H*W, H*W),
                                   nn.ReLU())
        
                
        self.G_conv = nn.Sequential(nn.Dropout(p = self.dropout),
                                    nn.Conv2d(C, 8, kernel_size=5, padding=2),
                                    nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.Dropout(p = self.dropout),
                                    nn.Conv2d(8, 4, kernel_size=5, padding=2),
                                    nn.BatchNorm2d(4),
                                    nn.ReLU(),
                                    nn.Dropout(p = self.dropout),
                                    nn.Conv2d(4, 2, kernel_size=5, padding=2),
                                    nn.BatchNorm2d(2),
                                    nn.ReLU(),
                                    nn.Dropout(p = self.dropout),
                                    nn.Conv2d(2, num_classes, kernel_size=5, 
                                              padding=2),
                                    nn.Softmax(dim = 1))
        
        if weight_scale:
            self.init_params()


    def forward(self, inputs):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        inputs = inputs.view(inputs.size(0), -1)
        out_lin = self.G_lin(inputs)
        out_lin = out_lin.view(inputs.size(0), *self.input_dim)
        out = self.G_conv(out_lin)
        
        return out


    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)
            
    def init_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

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


