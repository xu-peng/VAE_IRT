import torch
from torch import nn
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super(AutoEncoder, self).__init__()

        self.fc1 = nn.Linear(n_input, n_latent)
        self.fc2 = nn.Linear(n_latent, n_input)

    def encode(self, x):
        return self.fc1(x)

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return F.sigmoid(h)

    def forward(self, x):
        encoded = self.encode(x)
        return self.decode(encoded)