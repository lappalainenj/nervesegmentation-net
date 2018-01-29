import torch
import torch.nn as nn
from torch.nn import init

class NerveNET(nn.Module):

    def __init__(self, input_dim = (1, 128, 128), num_classes=2,
                 weight_scale = True, dropout = 0.1, leak = 0.01,
                 binary_out = True, upsample_unit = 'Upsample'):
        super(NerveNET, self).__init__()
        
        in_channels, W, H = input_dim
        self.binary_out = binary_out
        self.upsample_unit = upsample_unit
        self.dropout = dropout
        
        if num_classes > 1:
           activation_out = nn.Softmax(dim = 1)
        else:
            activation_out = nn.Sigmoid()
        
        assert upsample_unit in ['Upsample', 'ConvTranspose2d']
        if upsample_unit == 'Upsample':
            upsample_units = []
            upsample_units = [nn.Upsample(scale_factor = 2) for x in range(4)]
        elif upsample_unit == 'ConvTranspose2d':
            upsample_units = []
            upsample_units.append(nn.ConvTranspose2d(512,512,kernel_size=2,stride=2,padding=0))
            upsample_units.append(nn.ConvTranspose2d(256,256,kernel_size=2,stride=2,padding=0))
            upsample_units.append(nn.ConvTranspose2d(128,128,kernel_size=2,stride=2,padding=0))
            upsample_units.append(nn.ConvTranspose2d(64, 64,kernel_size=2,stride=2,padding=0))
        

        self.down1 = nn.Sequential(nn.Dropout(p = self.dropout),
                                   nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
                               
        self.down2 = nn.Sequential(nn.Dropout(p = self.dropout),
                                   nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        
        self.down3 = nn.Sequential(nn.Dropout(p = self.dropout),
                                   nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        
        self.down4 = nn.Sequential(nn.Dropout(p = self.dropout),
                                   nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        
        self.bottom = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(),
                                    upsample_units[0],
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())     
        
        W_b = int((W / 2**2 - 2) / 2 + 1)
        H_b = int((H / 2**2 - 2) / 2 + 1)
        
        self.binary = BinaryOut(W_b, H_b, activation_out, num_classes,
                                weight_scale, leak)
        
        self.up4 = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 upsample_units[1],
                                 nn.BatchNorm2d(256),
                                 nn.ReLU())
     
        
        self.up3 = nn.Sequential(nn.Conv2d(384, 128, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(128), 
                                 nn.ReLU(),
                                 upsample_units[2],
                                 nn.BatchNorm2d(128), 
                                 nn.ReLU())
        
        self.up2 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 upsample_units[3],
                                 nn.BatchNorm2d(64),
                                 nn.ReLU())
        
        self.out = nn.Sequential(nn.Conv2d(96, 32, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),
                                 nn.Conv2d(32, num_classes,kernel_size=3, padding=1),
                                 activation_out)
        
        self.pool = nn.MaxPool2d(2, 2)
                                 
        if weight_scale:
            self.init_params()


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        out = {}
        down1 = self.down1(x) #1 - 32
        down2 = self.down2(self.pool(down1)) #32 - 64
        down3 = self.down3(self.pool(down2)) #64 - 128
        down4 = self.down4(self.pool(down3)) #128 - 256
        
        bottom = self.bottom(self.pool(down4)) #256 - 512
        bottom_down4 = torch.cat([bottom, down4], dim = 1) #768
        up4 = self.up4(bottom_down4) #768 - 256
        up4_down3 = torch.cat([up4, down3], dim = 1) #384
        up3 = self.up3(up4_down3) #384 - 128
        up3_down2 = torch.cat([up3, down2], dim = 1) #192
        up2 = self.up2(up3_down2) #192 - 64 
        up2_down1 = torch.cat([up2, down1], dim = 1) #96
        
        out['main'] = self.out(up2_down1) #96 - 2
        
        if self.binary_out:
            out['binary'] = self.binary(bottom) 
        
        return out
                        
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)\
        or isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def init_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
            
    def init_weightscale(self, weight_scale):
        
        units = [self.down1,
                  self.down2,
                  self.down3,
                  self.down4,
                  self.binary.modules(),
                  self.bottom,
                  self.up4,
                  self.up3,
                  self.up2,
                  self.out]
        
        for _unit in units:
            
            for l in _unit:
                
                if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
                        l.weight.data.mul_(weight_scale)

                    
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

class BinaryOut(nn.Module):

    def __init__(self, W_b, H_b, activation_out, num_classes,
                 weight_scale, leak):
        
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(512, 512, kernel_size=5, padding=2),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU())
        
        if isinstance(activation_out, nn.Softmax):
            activation_out = nn.Softmax(dim = 0)
            
        self.lin = nn.Sequential(nn.Linear(W_b * H_b * 512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, num_classes),
                                 activation_out)
                
        if weight_scale:
            self.init_params()


    def forward(self, inputs):
        out = self.down(inputs).contiguous()
        out = self.lin(out.view(out.size()[0],-1))
        return out
    
    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def init_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


