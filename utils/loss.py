import torch
import torch.nn as nn
import torch.nn.functional as F


'''
For Anomaly Transformer
'''
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

'''
source from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
'''
class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, inputs, targets):
        '''
        inputs: 0 < p < 1
        targets: 0, 1
        '''
        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = p*targets + (1-p)*(1-targets)
        alpha_t = self.alpha * targets + (1 -self.alpha)*(1-targets)
        F_loss = alpha_t * ((1-p_t)**self.gamma) * ce_loss
        return F_loss.mean()


def soft_f1_loss(y_pred:torch.Tensor, y_true:torch.Tensor):
    tp = (y_true * y_pred).sum(dim=0)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)

    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)

    cost_class1 = 1 - soft_f1_class1
    cost_class0 = 1 - soft_f1_class0

    cost = 0.5 * (cost_class1 + cost_class0)
    return cost.mean()

