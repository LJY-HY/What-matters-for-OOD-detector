import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, Subset
from utils.utils import *

def cifar10(args, train_TF = None, test_TF = None):
    if train_TF is None and test_TF is None:
        train_TF = get_transform(args.in_dataset, 'train')
        test_TF = get_transform(args.in_dataset, 'test')
  
    train_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, transform = train_TF, download=True)
    test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=False, transform = test_TF, download=False)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
    if args.tuning:
        test_indices = list(range(len(test_dataset)))
        # val_dataset, test_dataset = random_split(test_dataset,[1000,9000],generator=torch.Generator().manual_seed(0))
        val_dataset, test_dataset = Subset(test_dataset, test_indices[:1000]), Subset(test_dataset, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, test_dataloader

def cifar100(args, train_TF = None, test_TF = None):
    if train_TF is None and test_TF is None:
        train_TF = get_transform(args.in_dataset, 'train')
        test_TF = get_transform(args.in_dataset, 'test')

    train_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, transform = train_TF, download=True)
    test_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=False, transform = test_TF, download=False)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
    if args.tuning:
        test_indices = list(range(len(test_dataset)))
        # val_dataset, test_dataset = random_split(test_dataset,[1000,9000],generator=torch.Generator().manual_seed(0))
        val_dataset, test_dataset = Subset(test_dataset, test_indices[:1000]), Subset(test_dataset, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, test_dataloader