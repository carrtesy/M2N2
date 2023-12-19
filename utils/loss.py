import torch
import torch.nn as nn
import torch.nn.functional as F


'''
For Anomaly Transformer
'''
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)