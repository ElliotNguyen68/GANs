import torchvision
from pathlib import WindowsPath
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from GANs import *
import matplotlib.pyplot as plt
# from utils import *
import time
torch.autograd.set_detect_anomaly(True)


def train(num_epochs, pre_G_dir=None, pre_D_dir=None):

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    noise_dim = 64  # 128,....
    img_dim = 28*28*1
    batch_size = 32
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ]
    )
    my_dataset = datasets.MNIST(
        root="dataset/", transform=transform, download=True)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
    disc = Discriminator()
    gene = Generater()

    if pre_D_dir and pre_G_dir:
        disc.load_state_dict(torch.load(pre_D_dir))
        gene.load_state_dict(torch.load(pre_G_dir))

    criterion = nn.BCELoss()
    lr = 3e-4
    optim_disc = torch.optim.Adam(disc.parameters(), lr=lr)
    optim_gene = torch.optim.Adam(gene.parameters(), lr=lr)

    writer_fake = SummaryWriter(f"logs/fake")
    writer_real = SummaryWriter(f"logs/real")

    fixed_noise = torch.randn((batch_size, noise_dim), device=device)

    for i in range(num_epochs):
        start = time.time()
        print(f"Epoch: {i} ")
        for batch_idx, (real, _) in enumerate(my_dataloader):
            real = real.view(-1, 784).to(device)
            noise = torch.rand((batch_size, noise_dim), device=device)
            # --------------------------
            # Train discriminator
            # --------------------------
            optim_disc.zero_grad()
            fake = gene(noise)
            loss_fake = criterion(disc(fake.detach()), torch.zeros(
                (batch_size, 1), device=device))
            loss_real = criterion(disc(real), torch.ones(
                (batch_size, 1), device=device))
            loss_disc = (loss_fake+loss_real)/2
            loss_disc.backward()
            optim_disc.step()
            # -------------------------
            # train generator
            # -------------------------
            # optim_gene.zero_grad()
            # optim_disc.zero_grad()

            # try to maximize prob disc incorrect
            loss_gene = criterion(disc(fake), torch.ones(
                (batch_size, 1), device=device))
            optim_gene.zero_grad()
            optim_disc.zero_grad()
            loss_gene.backward()
            optim_gene.step()

            # print(f"Batch {batch_idx}: {loss_disc} {loss_gene}")
        print(
            f"Loss D: {loss_disc:.4f}, loss G: {loss_gene:.4f}"
        )
        end = time.time()
        print(f"training time: {end-start :.4f} s ")
        # torch.save(disc.state_dict(), "fc_gan_disc.pth")
        # torch.save(gene.state_dict(), "fc_gan_gene.pth")

        with torch.no_grad():

            fake = gene(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image(
                "Mnist Fake Images", img_grid_fake, global_step=i
            )
            writer_real.add_image(
                "Mnist Real Images", img_grid_real, global_step=i
            )


if __name__ == "__main__":
    train(10)
    # noise = torch.rand(64, device=device)
    # gen = Generater().eval()
    # gen.load_state_dict(torch.load("fc_gan_gene.pth"))
    # with torch.no_grad():
    #     output = gen(noise).reshape(28, 28, 1).to("cpu")

    #     plt.imshow(output)
    #     plt.show()
