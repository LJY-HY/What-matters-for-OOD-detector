import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

cifar10_mean = (0.4914, 0.4823, 0.4466)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

def cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std = cifar10_std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std = cifar10_std)
    ])

    train_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, transform = transform_train, download=True)
    test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=False, transform = transform_test, download=False)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    return train_dataloader, test_dataloader

def cifar100(args):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)
    ])

    train_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, transform = transform_train, download=True)
    test_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=False, transform = transform_test, download=False)

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    return train_dataloader, test_dataloader