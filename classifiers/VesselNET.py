import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['VesselNET']


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x



class VesselNET(nn.Module):

    def __init__(self, num_classes=2):
        super(VesselNET, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 9, kernel_size=11, stride=1, padding=5),#128
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2), #63
            
            nn.Conv2d(9, 32, kernel_size=15, padding=7),#63
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2), #31          
            
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32*31*31, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 31 * 31)
        x = self.classifier(x)
        return x
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
