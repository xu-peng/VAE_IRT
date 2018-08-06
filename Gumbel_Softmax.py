# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca)
# The code has been modified from pytorch example vae code and inspired by the origianl tensorflow implementation of gumble-softmax by Eric Jang.

from __future__ import print_function
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import ipdb



class VAE_gumbel(nn.Module):

    def __init__(self, latent_dim, categorical_dim, temp):
        super(VAE_gumbel, self).__init__()

        self.fc1 = nn.Linear(9, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 9)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.temp = temp
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, latent_dim, categorical_dim):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        ipdb.set_trace()
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, latent_dim * categorical_dim)

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def gumbel(self, q):
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        z = self.gumbel_softmax(q_y, self.temp, self.latent_dim, self.categorical_dim)
        return z

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x):
        q = self.encode(x.view(-1, 9))
        z = self.gumbel(q)
        return self.decode(z), F.softmax(q)


# temp_min = 0.5
# ANNEAL_RATE = 0.00003
