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

    def __init__(self, latent_dim, categorical_dim):
        super(VAE_gumbel, self).__init__()

        self.fc1 = nn.Linear(9, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 9)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, latent_dim, categorical_dim, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
          Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
          Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            y_hard = (y_hard - y).detach() + y
            y = y_hard.view(-1, latent_dim * categorical_dim)
        return y

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def gumbel(self, q, temp):
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        z = self.gumbel_softmax(q_y, temp, self.latent_dim, self.categorical_dim)
        return z

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp):
        q = self.encode(x.view(-1, 9))
        z = self.gumbel(q, temp)
        return self.decode(z), F.softmax(q)


# temp_min = 0.5
# ANNEAL_RATE = 0.00003
