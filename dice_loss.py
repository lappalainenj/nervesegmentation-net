from torch.nn.modules.loss import _Loss

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class DiceLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(DiceLoss, self).__init__(size_average)
        self.reduce = reduce
        
    def forward(self, input, target):
        _assert_no_grad(target)
        
        assert target.size() == input.size(), "Target size different to input size."
        
        eps = 0.0000001
        p  = input.view(input.size()[0], -1).add(eps)
        gt = target.view(target.size()[0], -1).add(eps)
    
        numerator =  p.mul(gt).sum(1).mul(2)
        denumerator = p.mul(p).sum(1).add(gt.mul(gt).sum(1))
        
        if self.size_average:
            return (1 - numerator.div(denumerator)).mean()
        
        return (1 - numerator.div(denumerator)).sum()