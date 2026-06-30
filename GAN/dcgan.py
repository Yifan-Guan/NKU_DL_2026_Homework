import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
import torchvision.utils as vutils

from utils import show_imgs

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
results_dir = './results'
model_dir = './models'


class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.main(input)
        return out.view(-1, 1)


class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, nc=1, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.dim() == 2:
            input = input.view(input.size(0), input.size(1), 1, 1)
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def print_model():
    D = Discriminator()
    print(D)
    G = Generator()
    print(G)


def save_model(D, G):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(D.state_dict(), os.path.join(model_dir, 'DCGAN_D.pth'))
    torch.save(G.state_dict(), os.path.join(model_dir, 'DCGAN_G.pth'))


def load_generator(model_path=None, nz=100, ngpu=1, map_location=None):
    if model_path is None:
        model_path = os.path.join(model_dir, 'DCGAN_G.pth')
    if map_location is None:
        map_location = device

    generator = Generator(ngpu=ngpu, nz=nz).to(map_location)
    state_dict = torch.load(model_path, map_location=map_location)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def _to_noise_tensor(noise, nz=100, batch_size=None, device_override=None):
    if device_override is None:
        device_override = device

    if noise is None:
        if batch_size is None:
            batch_size = 8
        return torch.randn(batch_size, nz, device=device_override)

    noise_tensor = torch.as_tensor(noise, dtype=torch.float32, device=device_override)
    if noise_tensor.dim() == 1:
        return noise_tensor.view(1, -1)
    if noise_tensor.dim() == 4:
        return noise_tensor.view(noise_tensor.size(0), noise_tensor.size(1))
    return noise_tensor


def generate_images_from_noise(noise, model_path=None, save_path=None, nz=100, ngpu=1):
    generator = load_generator(model_path=model_path, nz=nz, ngpu=ngpu)
    noise_tensor = _to_noise_tensor(noise, nz=nz)
    if noise_tensor.size(1) != nz:
        raise ValueError(f'Expected noise dimension {nz}, got {noise_tensor.size(1)}')

    with torch.no_grad():
        fake = generator(noise_tensor)

    if save_path is None:
        save_path = os.path.join(results_dir, 'dcgan_custom_8.png')
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    show_imgs(fake, save_path=save_path)
    return fake


def generate_latent_traversal(base_noise=None, selected_indices=None, step_values=None,
                               model_path=None, save_path=None, nz=100, ngpu=1,
                               batch_size=8):
    if selected_indices is None:
        selected_indices = [0, 1, 2, 3, 4]
    if step_values is None:
        step_values = [-2.0, 0.0, 2.0]

    generator = load_generator(model_path=model_path, nz=nz, ngpu=ngpu)
    base_noise = _to_noise_tensor(base_noise, nz=nz, batch_size=batch_size)
    if base_noise.dim() != 2:
        raise ValueError('base_noise must be a 1D or 2D tensor-like object')
    if base_noise.size(1) != nz:
        raise ValueError(f'Expected base noise dimension {nz}, got {base_noise.size(1)}')

    rows = []
    row_labels = []
    with torch.no_grad():
        for latent_index in selected_indices:
            if latent_index < 0 or latent_index >= nz:
                raise ValueError(f'latent index {latent_index} is out of range for nz={nz}')
            for step_value in step_values:
                current_noise = base_noise.clone()
                current_noise[:, latent_index] = current_noise[:, latent_index] + float(step_value)
                rows.append(generator(current_noise))
                row_labels.append((latent_index, float(step_value)))

    fake = torch.cat(rows, dim=0)
    if save_path is None:
        save_path = os.path.join(results_dir, 'dcgan_latent_traversal_15x8.png')
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    show_imgs(fake, save_path=save_path)

    return {
        'fake': fake,
        'row_labels': row_labels,
        'save_path': save_path,
        'selected_indices': list(selected_indices),
        'step_values': list(step_values),
    }


def summarize_latent_traversal(row_labels):
    summary = []
    for latent_index, step_value in row_labels:
        summary.append(f'latent_dim_{latent_index}: step {step_value:+.2f}')
    return summary


def train(dataloader, epochs=3, nz=100, ngpu=1):
    criterion = nn.BCELoss()
    D = Discriminator(ngpu=ngpu).to(device)
    G = Generator(ngpu=ngpu, nz=nz).to(device)
    D.apply(weights_init)
    G.apply(weights_init)

    optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    all_lossD = []
    all_lossG = []
    collect_x_gen = []
    fixed_noise = torch.randn(64, nz, device=device)

    plt.figure()
    plt.ion()

    for epoch in range(epochs):
        current_lossD = 0.0
        current_lossG = 0.0

        pbar = tqdm.tqdm(enumerate(dataloader, 0), desc='Epoch {}/{}'.format(epoch + 1, epochs), total=len(dataloader))
        for _, data in pbar:
            x_real, _ = data
            x_real = x_real.to(device)
            batch_size = x_real.size(0)
            lab_real = torch.ones(batch_size, 1, device=device)
            lab_fake = torch.zeros(batch_size, 1, device=device)

            optimizerD.zero_grad()

            D_x = D(x_real)
            lossD_real = criterion(D_x, lab_real)

            z = torch.randn(batch_size, nz, device=device)
            x_gen = G(z).detach()
            D_G_z = D(x_gen)
            lossD_fake = criterion(D_G_z, lab_fake)

            lossD = lossD_real + lossD_fake
            current_lossD += lossD.item()
            lossD.backward()
            optimizerD.step()

            optimizerG.zero_grad()

            z = torch.randn(batch_size, nz, device=device)
            x_gen = G(z)
            D_G_z = D(x_gen)
            lossG = criterion(D_G_z, lab_real)
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
        show_imgs(x_gen, save_path=os.path.join(results_dir, f'dcgan_epoch_{idx}.png'))

    plt.figure()
    plt.plot(all_lossD, label='Discriminator loss')
    plt.plot(all_lossG, label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'dcgan_loss_plot.png'))

    save_model(D, G)
    print('Training finished. Model saved to', model_dir)