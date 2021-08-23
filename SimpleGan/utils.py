from numpy import sqrt
from torch._C import BenchmarkConfig
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import Normalize

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
noise_dim = 64  # 128,....
img_dim = 28*28*1
batch_size = 16
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)])
my_dataset = datasets.MNIST(
    root="dataset/", transform=transforms, download=True)


def Get_mean_std(data):
    n = len(data)
    sum = 0
    for x, _ in data:
        sum += torch.sum(x)/784
    mean = sum/n
    sum = 0
    for x, _ in data:
        sum += torch.sum(torch.square(x-mean))
    std = sqrt(sum/(784*n))
    return mean, std


print(my_dataset[0][0].shape)
# mean=0.1307
