import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Model

class IWAE(nn.Module):
    def __init__(self, d_data, d_latent, d_hidden=100):
        super(IWAE, self).__init__()
        self.d_latent = d_latent
        self.enc = Encoder(d_data, d_latent, d_hidden)
        self.dec = Decoder(d_data, d_latent, d_hidden)


    def loss(self, x, k):

        ######################################
        # TODO: implement k-sample IWAE loss #
        ######################################

        if "you just started, here's an autoencoder loss for you":
            mu, sigma = self.enc(x)
            z = mu
            x_r = self.dec(z)
            objective = torch.mean(bernoulliLL(x, x_r))
            loss = -objective

        return loss


def bernoulliLL(x, x_r):
    '''
    x: [B, Dx]
    x_r: [B, Dx]
    ---
    LL: [B]
    '''
    LL = torch.sum(x*torch.log(x_r)+(1-x)*torch.log(1-x_r), dim=1)
    return LL


class Encoder(nn.Module):
    def __init__(self, d_data, d_latent, d_hidden):
        super(Encoder, self).__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
            nn.Linear(d_data, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 2*d_latent)
        )

    def forward(self, x):
        out = self.layers(x)
        mu = torch.tanh(out[:, :self.d_latent])
        sigma = torch.exp(out[:, :self.d_latent])
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, d_data, d_latent, d_hidden):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_data),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_r = self.layers(z)
        return x_r



### Training Helpers

def get_data_loader(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)  
    return trainloader, testloader

def train_epoch(net, loader, k, optimizer):
    running_loss = 0.0
    running_n = 0
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        x_probs, y = batch
        x = torch.distributions.Bernoulli(x_probs).sample() # binarize
        x = x.to(device)
        B = x.size(0)
        x = x.view(B, -1)
        batch_loss = net.loss(x, k)
        batch_loss.backward()
        optimizer.step()
        running_n += B
        running_loss += B*batch_loss.detach().item()
    loss = running_loss/running_n
    return loss

def test_epoch(net, loader, k):
    running_loss = 0.0
    running_n = 0
    for batch_idx, batch in enumerate(loader):
        x_probs, y = batch
        x = torch.distributions.Bernoulli(x_probs).sample() # binarize
        x = x.to(device)
        B = x.size(0)
        x = x.view(B, -1)
        batch_loss = net.loss(x, k)
        running_n += B
        running_loss += B*batch_loss.detach().item()
    loss = running_loss/running_n
    return loss


### Visualizer

def make_z_grid(Dz, N, limit=1.0):
    if Dz == 2:
        # coordinate grid
        z_grid = torch.zeros(N, N, Dz)
        linsp = torch.linspace(-limit, limit, N)
        z_grid[:, :, 0] = linsp.view(-1, 1)
        z_grid[:, :, 1] = linsp.view(1, -1)
    else:
        # sample randomly
        z_grid = torch.randn(N, N, Dz)

    return z_grid


def show_img_grid(net, z_grid):
    N, _, Dz = z_grid.size()
    z_batch = z_grid.view(N*N, Dz).to(device)
    dec_img = net.dec(z_batch).cpu().detach().view(N*N, 28, 28)
    img_grid = dec_img.view(N, N, 28, 28).numpy()
    img_cat = np.concatenate(np.concatenate(img_grid, axis=2), axis=0)
    plt.imshow(img_cat)



### Main routine

def main(args):

    trainloader, testloader = get_data_loader(args.batch_size)
    net = IWAE(28*28, args.d_latent)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    if args.visualize:
        plt.ion()
        plt.show()
        z_grid = make_z_grid(args.d_latent, 8, limit=1.0)
    for epoch in range(args.epochs):
        train_loss = train_epoch(net, trainloader, args.train_k, optimizer)
        print("(Epoch %d) train loss : %.3f"%(epoch+1, train_loss))
        if args.visualize:
            show_img_grid(net, z_grid)
            plt.draw()
            plt.pause(0.001)
    test_loss = test_epoch(net, testloader, args.test_k)
    print("Estimated test NLL : %.3f"%(test_loss))
    print("(This evaluation is not valid until you implement IWAE)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IWAE')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--d_latent',type=int,default=10)
    parser.add_argument('--train_k',type=int,default=1) # set train_k > 1 for IWAE
    parser.add_argument('--test_k',type=int,default=100)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--visualize',type=bool,default=True)
    args = parser.parse_args()
    main(args)
