from torch.nn.modules.loss import _Loss
import numpy as np

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class DiceLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, classweights = [1, 1]):
        super(DiceLoss, self).__init__(size_average)
        self.reduce = reduce
        self.cw = classweights
        
    def forward(self, input, target):
        'Dice Loss calculated on Dualchannel Input and Singlechannel Target'
        _assert_no_grad(target)
        
        eps = 0.0000001
        
        Closses = 0
        for C in range(input.size(1)):
            Cinput = input[:, C].contiguous()
            target = target.squeeze()
            
            assert Cinput.size() == target.size(), "Target size different to input size."
            p  = Cinput.view(Cinput.size(0), -1).add(eps).float()
            gt = target.view(target.size(0), -1).add(eps).float()
            
            if C == 0:
                gt = 1 - gt
            numerator =  p.mul(gt).sum(1).mul(2)
            denumerator = p.mul(p).sum(1).add(gt.mul(gt).sum(1))
            if C == 0:
                Closses += self.cw[0]*(1-numerator.div(denumerator))
            else:                    
                Closses += self.cw[1]*(1 - numerator.div(denumerator))

        if self.size_average:
            return Closses.mean()
        return Closses.sum()
        
    
    def acc(self, input, target):
        'Dice Accuracy calculated on Singlechannel Input&Target'
        _assert_no_grad(target)
        
        eps = 0.0000001
        
        assert input.size() == target.size(), "Target size different to input size."
        p  = input.view(input.size(0), -1).add(eps)
        gt = target.view(target.size(0), -1).add(eps)
        
        numerator =  p.mul(gt).sum(1).mul(2)
        denumerator = p.mul(p).sum(1).add(gt.mul(gt).sum(1))
        
        if self.size_average:
            return (numerator/denumerator).mean()
        return (numerator/denumerator).sum()
    
    def dc_np(self, input, target):
        '''Just to check if numpy returns the same'''
        eps = 0.00000001
        gt = target.reshape(target.shape[0], -1) + eps
        p = input.reshape(input.shape[0], -1) + eps
        dice = np.sum(p[gt==1])*2.0 / (np.sum(p*p) + np.sum(gt*gt))
        return dice
        
        
        
        
        
            