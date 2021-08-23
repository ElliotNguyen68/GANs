import torch
from torch._C import device
import torch.nn as nn


device = torch.device("cuda")


class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28*1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.to(device)

    def forward(self, batch_img):
        return self.net(batch_img)


class Generater(nn.Module):
    def __init__(self, noise_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 784),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, noise):
        return self.net(noise)
