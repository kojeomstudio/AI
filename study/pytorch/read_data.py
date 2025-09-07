import torch
import torchvision
import torchvision.transforms as tr 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

transf = tr.Compose([tr.Resize(16), tr.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root='./study/pytorch/pytorch_data', train=True, download=True, transform=transf)
test_set = torchvision.datasets.CIFAR10(root='./study/pytorch/pytorch_data', train=False, download=True, transform=transf)

#print(train_set[0][0].size())

train_loader = DataLoader(train_set, batch_size=50, shuffle=True)
test_loader = DataLoader(test_set, batch_size=50, shuffle=False)

print(f"len(train_loader): {len(train_loader)}, len(test_loader): {len(test_loader)}")

images, labels = next(iter(train_loader))
print(f"images.size(): {images.size()}")

