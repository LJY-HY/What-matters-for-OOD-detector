import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split

cifar10_mean = (0.4914, 0.4823, 0.4466)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

def LSUN(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'LSUN_resize'))
    testsetout = datasets.ImageFolder(root=dataroot, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return 1, test_loader

def TinyImagenet(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
    ])
    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'Imagenet_resize'))
    testsetout = datasets.ImageFolder(dataroot, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)

    return 1, test_loader