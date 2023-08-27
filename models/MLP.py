'''
Basic MLP implementation by:
Dongmin Kim (tommy.dm.kim@gmail.com)
'''

import torch
import torch.nn as nn
from models.Normalizer import Detrender


class MLP(nn.Module):
    def __init__(self, seq_len, num_channels, latent_space_size, gamma, normalization="None",
                 use_sigmoid_output=False, use_dropout=False, use_batchnorm=False):
        super().__init__()
        self.L, self.C = seq_len, num_channels
        self.encoder = Encoder(seq_len*num_channels, latent_space_size, use_dropout, use_batchnorm)
        self.decoder = Decoder(seq_len*num_channels, latent_space_size, use_dropout, use_batchnorm)
        self.normalization = normalization

        if self.normalization == "Detrend":
            self.use_normalizer = True
            self.normalizer = Detrender(num_channels, gamma=gamma)
        else:
            self.use_normalizer = False

        self.use_sigmoid_output = use_sigmoid_output
        if self.use_sigmoid_output:
            self.sigmoid = torch.nn.Sigmoid()



    def forward(self, X):
        B, L, C = X.shape
        assert (L == self.L) and (C == self.C)

        if self.use_normalizer:
            X = self.normalizer(X, "norm")
        z = self.encoder(X.reshape(B, L*C))
        out = self.decoder(z).reshape(B, L, C)

        if self.use_sigmoid_output:
            out = self.sigmoid(out)
        if self.use_normalizer:
            out = self.normalizer(out, "denorm")
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, latent_space_size, use_dropout=False, use_batchnorm=False):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 4, latent_space_size)
        self.relu3 = nn.ReLU()


        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout1 = nn.Dropout(p=0.2)
            self.dropout2 = nn.Dropout(p=0.2)
            self.dropout3 = nn.Dropout(p=0.2)

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm1 = nn.BatchNorm1d(input_size//2)
            self.batchnorm2 = nn.BatchNorm1d(input_size//4)
            self.batchnorm3 = nn.BatchNorm1d(latent_space_size)


    def forward(self, x):
        x = self.linear1(x)
        if self.use_batchnorm:
            x = self.batchnorm1(x)
        x = self.relu1(x)
        if self.use_dropout:
            x = self.dropout1(x)

        x = self.linear2(x)
        if self.use_batchnorm:
            x = self.batchnorm2(x)
        x = self.relu2(x)
        if self.use_dropout:
            x = self.dropout2(x)

        x = self.linear3(x)
        if self.use_batchnorm:
            x = self.batchnorm3(x)
        x = self.relu3(x)
        if self.use_dropout:
            x = self.dropout3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, latent_space_size, use_dropout=False, use_batchnorm=False):
        super().__init__()
        self.linear1 = nn.Linear(latent_space_size, input_size // 4)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 4, input_size // 2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 2, input_size)

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout1 = nn.Dropout(p=0.2)
            self.dropout2 = nn.Dropout(p=0.2)
            self.dropout3 = nn.Dropout(p=0.2)

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm1 = nn.BatchNorm1d(input_size//4)
            self.batchnorm2 = nn.BatchNorm1d(input_size//2)

    def forward(self, x):
        x = self.linear1(x)
        if self.use_batchnorm:
            x = self.batchnorm1(x)
        x = self.relu1(x)
        if self.use_dropout:
            x = self.dropout1(x)

        x = self.linear2(x)
        if self.use_batchnorm:
            x = self.batchnorm2(x)
        x = self.relu2(x)
        if self.use_dropout:
            x = self.dropout2(x)

        out = self.linear3(x)
        return out



