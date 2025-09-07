import torch
import torchvision
import torchvision.transforms as tr 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

transf = tr.Compose([tr.Resize(16), tr.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root='./pytorch_data', train=True, download=True, transform=transf)
test_set = torchvision.datasets.CIFAR10(root='./pytorch_data', train=False, download=True, transform=transf)

print(train_set[0][0].size())