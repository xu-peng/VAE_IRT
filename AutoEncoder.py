import torch
from torch import nn
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                      padding=0, dilation=1, groups=1, bias=True),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

#            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride=1,
#                      padding=0, dilation=1, groups=1, bias=True),
#            nn.Sigmoid()
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1,
                               padding=0, dilation=1, groups=1, bias=True),
            # nn.BatchNorm1d(256),
            nn.Sigmoid(),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=1,
                               padding=0, dilation=1, groups=1, bias=True),
            # nn.BatchNorm1d(256),
            nn.Sigmoid(),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=1,
                               padding=0, dilation=1, groups=1, bias=True),
            # nn.BatchNorm1d(256),
            nn.Sigmoid(),
        )

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=3, stride=1,
                               padding=0, dilation=1, groups=1, bias=True),
            # nn.BatchNorm1d(256),
            nn.Sigmoid(),
        )

        self.shrink = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1, stride=1,
                      padding=0, dilation=1, groups=1, bias=True),
            # nn.BatchNorm1d(256),
            nn.Sigmoid(),
        )


    def encode(self, x):
        return self.encoder(x)

    def decode1(self, z):
        return self.decoder1(z)

    def decode2(self, z):
        return self.decoder2(z)

    def decode3(self, z):
        return self.decoder3(z)

    def decode4(self, z):
        return self.decoder4(z)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode1(encoded)
        decoded = self.decode2(decoded)
        decoded = self.decode3(decoded)
        return self.decode4(decoded)
