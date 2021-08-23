from os import abort
import torch
import torch.nn as nn
from torch.nn.functional import linear
from torch.nn.modules.activation import LeakyReLU
a = torch.rand((5, 1, 32, 32))
conv = nn.Conv2d(1, 32, kernel_size=4, stride=2)
print(conv(a).shape)
conved_a = conv(a)
upconv = nn.ConvTranspose2d(
    in_channels=32, out_channels=1, kernel_size=4, stride=2)

b_norm = nn.BatchNorm2d(1)

upconved = upconv(conved_a)
print(upconved.shape)
print(upconved)
print("*******************************")
normed = b_norm(upconved)
flat = normed.squeeze_(1).view(5, -1)
print(torch.mean(flat, dim=1))
print(normed.shape)
print("daf \n adf")
# print(upconv(conved_a).shape)
# print(a)
