import torch
import torch.nn as nn


class Detrender(nn.Module):
    def __init__(self, num_features: int, gamma=0.99):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        """
        super(Detrender, self).__init__()
        self.num_features = num_features
        self.gamma = gamma
        self.mean = nn.Parameter(torch.zeros(1, 1, self.num_features), requires_grad=False)


    def forward(self, x, mode:str):
        if mode == 'norm':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x


    def _update_statistics(self, x):
        dim2reduce = tuple(range(0, x.ndim-1))
        mu = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.mean.lerp_(mu, 1-self.gamma)


    def _set_statistics(self, x:torch.Tensor):
        self.mean = nn.Parameter(x, requires_grad=False)


    def _normalize(self, x):
        x = x - self.mean
        return x


    def _denormalize(self, x):
        x = x + self.mean
        return x