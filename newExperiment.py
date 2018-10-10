import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import copy
from tensorboardX import SummaryWriter

P = np.array(np.random.rand(400, 3) > .5)
Q1 = np.array(np.random.rand(3, 6) > .5)
Q2 = np.array([[True, False, False], [False, True, False], [False, False, True]])
Q = np.concatenate((Q1, Q2), axis=1)
R = np.dot(P, Q).astype(np.float32)

R_true = copy.copy(R)
# Choose 20% data to be noise, that is 720 of 3600 entries
index = np.random.choice(3600, 720, replace=False)
rowIndex = (index/9).astype(int)
colIndex = index % 9
# inject noise
R[rowIndex, colIndex] = 1 - R[rowIndex, colIndex]

# Check:
# (R_true == R).sum()  = 3519
# (R_true[rowIndex, colIndex] != R[rowIndex, colIndex]).sum()=720


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(9, 400),
            nn.ReLU(True),
            nn.Linear(400, 3),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(3, 400),
            nn.ReLU(True),
            nn.Linear(400, 9),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(9, 400)
        self.fc21 = nn.Linear(400, 3)
        self.fc22 = nn.Linear(400, 3)
        self.fc3 = nn.Linear(3, 400)
        self.fc4 = nn.Linear(400, 9)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 9))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DCVAE(nn.Module):
    def __init__(self):
        super(DCVAE, self).__init__()

        self.conv_mu = nn.Conv1d(128, 20, 1)
        self.conv_logvar = nn.Conv1d(128, 20, 1)

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Tanh())

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(20, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # input is (nc) x 64 x 64
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

    def encode(self, x):
        output = self.encoder(x)
        return [self.conv_mu(output), self.conv_logvar(output)]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.decoder(z)
        return h3

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



device = "cpu"
model = DCVAE().to(device)
# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.SGD(model.parameters(), lr=1e-5)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD


batch_size = 8

writer = SummaryWriter()

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(int(len(R)/batch_size)):
        data = torch.tensor(R[batch_idx*batch_size:((batch_idx+1)*batch_size), :])
        data = data.to(device)
        data = data.view(8, 1, 9)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        BCE, KLD = loss_function(recon_batch, data, mu, logvar)
        loss = BCE
        # loss = BCE + KLD
        # recon_batch = model(data)
        # loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        writer.add_scalar('train_elbo', -train_loss, global_step=epoch + 1)
        writer.add_scalar('train_kl', KLD/len(data), global_step=epoch + 1)
        writer.add_scalar('train_bce', BCE/len(data), global_step=epoch + 1)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t BCE: {:.2f}\tKLD: {:.2f}\tLoss: {:.2f}'.format(
             epoch, (batch_idx+1) * batch_size, len(R),
             100.0 * batch_size * (batch_idx+1) / len(R),
             BCE/len(data), KLD/len(data), loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(R)))


for epoch in range(1, 101):
    train(epoch)

recon, mu, logvar = model(torch.tensor(R).view(400, 1, 9))
recon = recon.view(400, 9)
(np.rint(recon.detach().numpy())==R_true).sum()


# Generate ideal response pattern
import itertools
lst = list(map(list, itertools.product([0, 1], repeat=3)))
profile = np.array(lst, dtype=bool)
R = np.dot(profile, Q).astype(np.float32)


model = autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(int(len(R)/batch_size)):
        data = torch.tensor(R[batch_idx*batch_size:((batch_idx+1)*batch_size), :])
        data = data.to(device)
        data = data.view(8, 1, 9)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = criterion(recon_batch, data)
        # recon_batch = model(data)
        # loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.2f}'.format(
             epoch, (batch_idx+1) * batch_size, len(R),
             100.0 * batch_size * (batch_idx+1) / len(R),
             loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(R)))


for epoch in range(1, 1001):
    train(epoch)

recon = model(torch.tensor(R).view(400, 1, 9)).view(400, 9)
(np.rint(recon.detach().numpy()) == R_true).sum()
np.clip()