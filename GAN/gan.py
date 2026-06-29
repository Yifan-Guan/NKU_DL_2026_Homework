import os
import tqdm
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from utils import show_imgs

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
results_dir = './results'
model_dir = './models'

class Discriminator(torch.nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), 784) # flatten (bs x 1 x 28 x 28) -> (bs x 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)

    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out) # range [-1, 1]
        # convert to image 
        out = out.view(out.size(0), 1, 28, 28)
        return out

def print_model():
    D = Discriminator()
    print(D)
    G = Generator()
    print(G)

def save_model(D, G):
    torch.save(D.state_dict(), os.path.join(model_dir, f'GAN_D.pth'))
    torch.save(G.state_dict(), os.path.join(model_dir, f'GAN_G.pth'))

def train(dataloader, epochs = 3):
    criterion = nn.BCELoss()
    # Re-initialize D, G:
    D = Discriminator().to(device)
    G = Generator().to(device)
    # Now let's set up the optimizers (Adam, better than SGD for this)
    optimizerD = torch.optim.SGD(D.parameters(), lr=0.03)
    optimizerG = torch.optim.SGD(G.parameters(), lr=0.03)
    # optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002)
    # optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002)
    lab_real = torch.ones(64, 1, device=device)
    lab_fake = torch.zeros(64, 1, device=device)

    all_lossD = []
    all_lossG = []

    # for logging:
    collect_x_gen = []
    fixed_noise = torch.randn(64, 100, device=device)
    fig = plt.figure() # keep updating this one
    plt.ion()

    for epoch in range(epochs): # 3 epochs
        current_lossD = 0.0
        current_lossG = 0.0

        pbar = tqdm.tqdm(enumerate(dataloader, 0), desc='Epoch {}/{}'.format(epoch, epochs), total=len(dataloader))
        for i, data in pbar:
            # STEP 1: Discriminator optimization step
            x_real, _ = next(iter(dataloader))
            x_real = x_real.to(device)
            # reset accumulated gradients from previous iteration
            optimizerD.zero_grad()

            D_x = D(x_real)
            lossD_real = criterion(D_x, lab_real)

            z = torch.randn(64, 100, device=device) # random noise, 64 samples, z_dim=100
            x_gen = G(z).detach()
            D_G_z = D(x_gen)
            lossD_fake = criterion(D_G_z, lab_fake)

            lossD = lossD_real + lossD_fake
            current_lossD += lossD.item()
            lossD.backward()
            optimizerD.step()

            # STEP 2: Generator optimization step
            # reset accumulated gradients from previous iteration
            optimizerG.zero_grad()

            z = torch.randn(64, 100, device=device) # random noise, 64 samples, z_dim=100
            x_gen = G(z)
            D_G_z = D(x_gen)
            lossG = criterion(D_G_z, lab_real) # -log D(G(z))
            current_lossG += lossG.item()

            lossG.backward()
            optimizerG.step()

        x_gen = G(fixed_noise)
        collect_x_gen.append(x_gen.detach().clone())
        current_lossD /= len(dataloader)
        current_lossG /= len(dataloader)
        all_lossD.append(current_lossD)
        all_lossG.append(current_lossG)

    for idx, x_gen in enumerate(collect_x_gen):
        show_imgs(x_gen, save_path=os.path.join(results_dir, f'epoch_{idx}.png'))

    plt.figure()
    plt.plot(all_lossD, label='Discriminator loss')
    plt.plot(all_lossG, label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_plot.png'))

    save_model(D, G)
    print('Training finished. Model saved to', model_dir)