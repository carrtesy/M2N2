'''
Basic MLP implementation by:
Dongmin Kim (tommy.dm.kim@gmail.com)
'''

import torch
import torch.nn as nn
from models.Normalizer import RevIN, SlowRevIN


class MLP(nn.Module):
    def __init__(self, seq_len, num_channels, latent_space_size, gamma, normalization="None"):
        super().__init__()
        self.L, self.C = seq_len, num_channels
        self.encoder = Encoder(seq_len*num_channels, latent_space_size)
        self.decoder = Decoder(seq_len*num_channels, latent_space_size)
        self.normalization = normalization

        if self.normalization == "RevIN":
            self.use_normalizer = True
            self.normalizer = RevIN(num_channels)
        elif self.normalization == "SlowRevIN":
            self.use_normalizer = True
            self.normalizer = SlowRevIN(num_channels, gamma=gamma)
        else:
            self.use_normalizer = False


    def forward(self, X):
        B, L, C = X.shape
        assert (L == self.L) and (C == self.C)

        if self.use_normalizer:
            X = self.normalizer(X, "norm")
        z = self.encoder(X.reshape(B, L*C))
        out = self.decoder(z).reshape(B, L, C)
        if self.use_normalizer:
            out = self.normalizer(out, "denorm")
        return out


class Encoder(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size // 2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 2, input_size // 4)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 4, latent_space_size)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        out = self.relu3(x)
        return out


class Decoder(nn.Module):
    def __init__(self, input_size, latent_space_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_space_size, input_size // 4)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_size // 4, input_size // 2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_size // 2, input_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        out = self.linear3(x)
        return out



