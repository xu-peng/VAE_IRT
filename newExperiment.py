import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import copy

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

device = "cpu"
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 9), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE, KLD


batch_size = 8


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
        loss = BCE + KLD
        # recon_batch = model(data)
        # loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t BCE: {:.2f}\tKLD: {:.2f}\tLoss: {:.2f}'.format(
             epoch, (batch_idx+1) * batch_size, len(R),
             100.0 * batch_size * (batch_idx+1) / len(R),
             BCE/len(data), KLD/len(data), loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(R)))


for epoch in range(1, 101):
    train(epoch)



recon, mu, logvar = model(torch.tensor(R).view(400,1,9))
(np.rint(recon.detach().numpy())==R_true).sum()