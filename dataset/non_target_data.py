import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, Subset

cifar10_mean = (0.4914, 0.4823, 0.4466)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

def LSUN(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
    ])
    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'LSUN_resize'))
    testsetout = datasets.ImageFolder(root=dataroot, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)
    if args.tuning:
        # val_dataset, test_dataset = random_split(testsetout,[1000,9000],generator=torch.Generator().manual_seed(0))# 이게 1000,9000으로 나누어지는지 확인해야함
        test_indices = list(range(len(testsetout)))
        val_dataset, test_dataset = Subset(testsetout, test_indices[:1000]), Subset(testsetout, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_loader

def LSUN_FIX(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
    ])
    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'LSUN_pil'))
    testsetout = datasets.ImageFolder(root=dataroot, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)
    if args.tuning:
        # val_dataset, test_dataset = random_split(testsetout,[1000,9000],generator=torch.Generator().manual_seed(0))# 이게 1000,9000으로 나누어지는지 확인해야함
        test_indices = list(range(len(testsetout)))
        val_dataset, test_dataset = Subset(testsetout, test_indices[:1000]), Subset(testsetout, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_loader

def TinyImagenet(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
    ])
    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'Imagenet_resize'))
    testsetout = datasets.ImageFolder(dataroot, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)
    if args.tuning:
        # val_dataset, test_dataset = random_split(testsetout,[1000,9000],generator=torch.Generator().manual_seed(0))
        test_indices = list(range(len(testsetout)))
        val_dataset, test_dataset = Subset(testsetout, test_indices[:1000]), Subset(testsetout, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_loader

def TinyImagenet_FIX(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
    ])
    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'Imagenet_pil'))
    testsetout = datasets.ImageFolder(dataroot, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)
    if args.tuning:
        # val_dataset, test_dataset = random_split(testsetout,[1000,9000],generator=torch.Generator().manual_seed(0))
        test_indices = list(range(len(testsetout)))
        val_dataset, test_dataset = Subset(testsetout, test_indices[:1000]), Subset(testsetout, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_loader