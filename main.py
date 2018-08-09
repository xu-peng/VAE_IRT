from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
# from AutoEncoder import AutoEncoder
# import VAE
# import Gumbel_Softmax
from Gumbel_Softmax import VAE_gumbel


parser = argparse.ArgumentParser(description='VAE_IRT')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Generate synthetic dataset

N = 400
D = 20
L = 9
R = np.array(np.random.rand(N, L) > .5, dtype=np.float32)

P = np.array(np.random.rand(400, 3) > .5)
Q1 = np.array(np.random.rand(3, 6) > .5)
Q2 = np.array([[True, False, False], [False, True, False], [False, False, True]])
Q = np.concatenate((Q1, Q2), axis=1)
R = np.dot(P, Q).astype(np.float32)

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)



# model = VAE().to(device)
# model = AutoEncoder().to(device)
model = VAE_gumbel(latent_dim=1, categorical_dim=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=1e-5)

# Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x):
#    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # MSE = (recon_x - x).pow(2).sum()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#    return BCE


def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 9), size_average=False)

    log_qy = torch.log(qy + 1e-20)
    g = torch.log(torch.Tensor([1.0 / 3])).cuda()
    KLD = -torch.sum(qy * (log_qy - g))

    return BCE + KLD

batch_size = args.batch_size
# temp = 1.0
temp_min = 0.1
ANNEAL_RATE = 0.03


def train(epoch):
    model.train()
    train_loss = 0
    temp = 10.0
    for batch_idx in range(int(len(R)/batch_size)):
        data = torch.tensor(R[batch_idx*batch_size:((batch_idx+1)*batch_size), :])
        data = data.to(device)
        data = data.view(batch_size, 1, 9, 1)
        optimizer.zero_grad()
        recon_batch, qy = model(data, temp)
        loss = loss_function(recon_batch, data, qy)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
             epoch, (batch_idx+1) * batch_size, len(R),
             100.0 * batch_size * (batch_idx+1) / len(R),
             loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(R)))


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    # test_loss /= len(test_loader.dataset)
    # print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
#     test(epoch)
#     with torch.no_grad():
#         sample = torch.randn(64, 20).to(device)
#         sample = model.decode(sample).cpu()
#         save_image(sample.view(64, 1, 28, 28),
#                    'results/sample_' + str(epoch) + '.png')

q = model.encode(torch.tensor(R[0:batch_size,:]).view(batch_size, 1, 9).to(device))
z = model.gumbel(q, 0.1)
recon = model.decode(z)

(torch.tensor(R[0:batch_size,:]).view(batch_size, 1, 9) - recon).pow(2).sum()

1 - torch.tensor(P[0:batch_size,:].astype(int)).to(device)