import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split, Subset
from utils.utils import *

def LSUN(args):
    train_TF = get_transform(args.in_dataset, 'train')
    test_TF = get_transform(args.in_dataset, 'test')

    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'LSUN_resize'))
    testsetout = datasets.ImageFolder(root=dataroot, transform=test_TF)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)
    if args.tuning:
        # val_dataset, test_dataset = random_split(testsetout,[1000,9000],generator=torch.Generator().manual_seed(0))# 이게 1000,9000으로 나누어지는지 확인해야함
        test_indices = list(range(len(testsetout)))
        val_dataset, test_dataset = Subset(testsetout, test_indices[:1000]), Subset(testsetout, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_loader

def LSUN_FIX(args):
    train_TF = get_transform(args.in_dataset, 'train')
    test_TF = get_transform(args.in_dataset, 'test')

    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'LSUN_pil'))
    testsetout = datasets.ImageFolder(root=dataroot, transform=test_TF)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)
    if args.tuning:
        # val_dataset, test_dataset = random_split(testsetout,[1000,9000],generator=torch.Generator().manual_seed(0))# 이게 1000,9000으로 나누어지는지 확인해야함
        test_indices = list(range(len(testsetout)))
        val_dataset, test_dataset = Subset(testsetout, test_indices[:1000]), Subset(testsetout, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_loader

def TinyImagenet(args):
    train_TF = get_transform(args.in_dataset, 'train')
    test_TF = get_transform(args.in_dataset, 'test')

    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'Imagenet_resize'))
    testsetout = datasets.ImageFolder(dataroot, transform=test_TF)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)
    if args.tuning:
        # val_dataset, test_dataset = random_split(testsetout,[1000,9000],generator=torch.Generator().manual_seed(0))
        test_indices = list(range(len(testsetout)))
        val_dataset, test_dataset = Subset(testsetout, test_indices[:1000]), Subset(testsetout, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_loader

def TinyImagenet_FIX(args):
    train_TF = get_transform(args.in_dataset, 'train')
    test_TF = get_transform(args.in_dataset, 'test')
    
    dataroot = os.path.expanduser(os.path.join('/home/esoc/repo/datasets/pytorch/', 'Imagenet_pil'))
    testsetout = datasets.ImageFolder(dataroot, transform=test_TF)
    test_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size, shuffle=False, num_workers=1)
    if args.tuning:
        # val_dataset, test_dataset = random_split(testsetout,[1000,9000],generator=torch.Generator().manual_seed(0))
        test_indices = list(range(len(testsetout)))
        val_dataset, test_dataset = Subset(testsetout, test_indices[:1000]), Subset(testsetout, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_loader