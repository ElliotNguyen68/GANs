import torch
from torch._C import device
import torch.nn as nn
import torch
from torch.nn.modules.activation import Sigmoid

device = torch.device("cuda:0")


class Discriminator(nn.Module):
    def __init__(self, img_channel):
        super().__init__()
        self.net = self.block(img_channel)
        self.to(device)

    def block(self, img_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Generator(nn.Module):
    def __init__(self, latent_space_dim, img_chanel):
        super().__init__()
        self.latent_size = latent_space_dim
        self.net = self.block(latent_space_dim, img_chanel)
        self.img_chanel = img_chanel
        self.to(device)

    def block(self, latent_space_dim, img_chanel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_space_dim,
                               out_channels=512, kernel_size=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256,
                               kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128,
                               kernel_size=3, stride=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=img_chanel, kernel_size=4, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.latent_size, 1, 1)
        return self.net(x).view(batch_size, self.img_chanel, 28, 28)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


disc = Discriminator(1)
img = torch.rand((5, 1, 28, 28), device=device)
print(disc(img).shape)
gen = Generator(32, 1)
noise = torch.rand((5, 32, 1, 1), device=device)
print(gen(noise).shape)
print(disc.count_parameters())
print(gen.count_parameters())
