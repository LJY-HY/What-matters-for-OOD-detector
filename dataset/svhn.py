import numpy as np
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
svhn_mean = (129.3/255, 124.1/255, 112.4/255)
svhn_std = (68.2/255, 65.4/255.0, 70.4/255.0)

cifar10_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std = cifar10_std)
    ])

cifar10_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std = cifar10_std)
    ])

cifar100_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std = cifar100_std)
    ])

cifar100_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std = cifar100_std)
    ])

svhn_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std = svhn_std)
    ])

svhn_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std = svhn_std)
    ])

def svhn(args):
    if args.in_dataset == 'cifar10':
        train_TF = cifar10_train_transform
        test_TF = cifar10_test_transform
    elif args.in_dataset == 'cifar100':
        train_TF = cifar100_train_transform
        test_TF = cifar100_test_transform
    elif args.in_dataset =='svhn':
        train_TF = svhn_train_transform
        test_TF = svhn_test_transform

    train_dataset = datasets.SVHN(root = '/home/esoc/repo/datasets/pytorch/svhn', split = 'train', transform = train_TF, download=True)
    test_dataset = datasets.SVHN(root = '/home/esoc/repo/datasets/pytorch/svhn', split = 'test', transform = test_TF, download=True)
    # test_10000_dataset, _ = random_split(test_dataset,[10000,16032],generator=torch.Generator().manual_seed(0))
    test_indices = list(range(len(test_dataset)))
    test_10000_dataset = Subset(test_dataset, test_indices[:10000])

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_10000_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    if args.tuning:
        # val_dataset, test_dataset = random_split(test_10000_dataset,[1000,9000],generator=torch.Generator().manual_seed(0))
        test_indices = list(range(len(test_10000_dataset)))
        val_dataset, test_dataset = Subset(test_10000_dataset, test_indices[:1000]), Subset(test_10000_dataset, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, test_dataloader