import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split

cifar10_mean = (0.4914, 0.4823, 0.4466)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
svhn_mean = (129.3/255, 124.1/255, 112.4/255)
svhn_std = (68.2/255, 65.4/255.0, 70.4/255.0)

def svhn(args):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125)),
        transforms.ToTensor(),
        transforms.Normalize(mean = svhn_mean, std = svhn_std)
        # transforms.Normalize(mean=cifar100_mean, std = cifar10_std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = svhn_mean, std = svhn_std)
        # transforms.Normalize(mean=cifar100_mean, std = cifar10_std)
    ])

    train_dataset = datasets.SVHN(root = '/home/esoc/repo/datasets/pytorch/svhn', split = 'train', transform = transform_train, download=True)
    test_dataset = datasets.SVHN(root = '/home/esoc/repo/datasets/pytorch/svhn', split = 'test', transform = transform_test, download=True)
    test_10000_dataset, _ = random_split(test_dataset,[10000,16032])

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    test_dataloader = DataLoader(test_10000_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    return train_dataloader, test_dataloader