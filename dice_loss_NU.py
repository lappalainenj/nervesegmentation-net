from torch.nn.modules.loss import _Loss

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
        _assert_no_grad(target)
        
        c = input.size(1)
        eps = 0.0000001
           
        if c > 1:
            
            Closses = 0
            for C in range(c):
                Cinput = input[:, C].contiguous()
                p  = Cinput.view(Cinput.size()[0], -1).add(eps).float()
                gt = target.view(target.size()[0], -1).add(eps).float()
                assert gt.size() == p.size(), "Target size different to input size."
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
        
        p  = input.view(input.size()[0], -1).add(eps)
        gt = target.view(target.size()[0], -1).add(eps)
        assert gt.size() == p.size(), "Target size different to input size."

        numerator =  p.mul(gt).sum(1).mul(2)
        denumerator = p.mul(p).sum(1).add(gt.mul(gt).sum(1))
        
        if self.size_average:
            return (1-numerator/denumerator).mean()
        return (1-numerator/denumerator).sum()
        
        
        
        
        
            