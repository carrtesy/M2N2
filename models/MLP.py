'''
Basic MLP implementation by:
Dongmin Kim (tommy.dm.kim@gmail.com)
'''

import torch
import torch.nn as nn
from models.RevIN import RevIN, ARevIN

class MLP(nn.Module):
    def __init__(self, seq_len, num_channels, latent_space_size, gamma, RevIN="None"):
        super().__init__()
        self.L, self.C = seq_len, num_channels
        self.encoder = Encoder(seq_len*num_channels, latent_space_size)
        self.decoder = Decoder(seq_len*num_channels, latent_space_size)
        self.RevIN = RevIN
        if self.RevIN != "None":
            self.use_RevIN = True
            self.revin = RevIN(num_channels) if self.RevIN == "RevIN" else ARevIN(num_channels, gamma=gamma)
        else:
            self.use_RevIN = False


    def forward(self, X):
        B, L, C = X.shape
        assert (L == self.L) and (C == self.C)

        if self.use_RevIN:
            X = self.revin(X, "norm")
        z = self.encoder(X.reshape(B, L*C))
        out = self.decoder(z).reshape(B, L, C)
        if self.use_RevIN:
            out = self.revin(out, "denorm")
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



