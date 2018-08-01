import torch
from torch import nn
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),

            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),

            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, stride=1,
                      padding=0, dilation=1, groups=1, bias=True),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                               padding=0, dilation=1, groups=1, bias=True),
            # nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                               padding=0, dilation=1, groups=1, bias=True),
            # nn.BatchNorm1d(64),
            nn.Sigmoid(),

            nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                               padding=0, dilation=1, groups=1, bias=True),
            # nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        return self.decode(encoded)
