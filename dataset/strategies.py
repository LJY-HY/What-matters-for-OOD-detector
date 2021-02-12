import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from utils.utils import *
from torch.utils.data import random_split, Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os.path
cifar10_mean = (0.4914, 0.4823, 0.4466)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
svhn_mean = (129.3/255, 124.1/255, 112.4/255)
svhn_std = (68.2/255, 65.4/255.0, 70.4/255.0)

cifar10_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std = cifar10_std),
    ])

cifar100_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std = cifar100_std),
    ])

svhn_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std = svhn_std),
    ])

cifar10_rot_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std = cifar10_std),
        Rotation()
    ])

cifar100_rot_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std = cifar100_std),
        Rotation()
    ])

svhn_rot_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std = svhn_std),
        Rotation()
    ])

cifar10_perm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std = cifar10_std),
        CutPerm()
    ])

cifar100_perm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std = cifar100_std),
        CutPerm()
    ])

svhn_perm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std = svhn_std),
        CutPerm()
    ])

def Aug_Rot(args):
    if args.in_dataset == 'cifar10':
        test_TF = cifar10_rot_transform
        test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=True, transform = test_TF, download=False)
    elif args.in_dataset == 'cifar100':
        test_TF = cifar100_rot_transform
        test_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=True, transform = test_TF, download=False)
    elif args.in_dataset =='svhn':
        test_TF = svhn_rot_transform
        test_dataset = datasets.SVHN(root = '/home/esoc/repo/datasets/pytorch/svhn', split = 'test', transform = test_TF, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    if args.tuning:
        test_indices = list(range(len(test_dataset)))
        val_dataset, test_dataset = Subset(test_dataset, test_indices[:1000]), Subset(test_dataset, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_dataloader


def Aug_Perm(args):
    if args.in_dataset == 'cifar10':
        test_TF = cifar10_perm_transform
        test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=False, transform = test_TF, download=False)
    elif args.in_dataset == 'cifar100':
        test_TF = cifar100_perm_transform
        test_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=False, transform = test_TF, download=False)
    elif args.in_dataset =='svhn':
        test_TF = svhn_perm_transform
        test_dataset = datasets.SVHN(root = '/home/esoc/repo/datasets/pytorch/svhn', split = 'test', transform = test_TF, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
    if args.tuning:
        test_indices = list(range(len(test_dataset)))
        val_dataset, test_dataset = Subset(test_dataset, test_indices[:1000]), Subset(test_dataset, test_indices[1000:])
        val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        return 1, val_dataloader, test_dataloader
    return 1, test_dataloader

def Adversarial(args):
    # Adversarial Samples are Based on FGSM attacked Samples
    clean_data_filename = args.outf+'clean_data_'+args.arch+'_'+args.in_dataset+'_Adversarial.pth'
    adv_data_filename = args.outf+'adv_data_'+args.arch+'_'+args.in_dataset+'_Adversarial.pth'
    noisy_data_filename = args.outf+'noisy_data_'+args.arch+'_'+args.in_dataset+'_Adversarial.pth'
    label_filename = args.outf+'label_'+args.arch+'_'+args.in_dataset+'_Adversarial.pth'
    if (os.path.isfile(clean_data_filename) &os.path.isfile(adv_data_filename) &os.path.isfile(noisy_data_filename) &os.path.isfile(label_filename)) is not True:
        net = globals()[args.arch](args).to(args.device)
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=4e-5)
        if args.path is not None:
            checkpoint = torch.load(args.path)
        else:
            checkpoint = torch.load('./checkpoint/'+args.in_dataset+'/'+args.arch)
        net.load_state_dict(checkpoint)
        net.eval()

        if args.arch in ['DenseNet']:
            min_pixel = -1.98888885975
            max_pixel = 2.12560367584
            if args.in_dataset == 'cifar10':
                random_nosie_size = 0.21/4
            elif args.in_dataset == 'cifar100':
                random_nosie_size = 0.21/8
            else:
                random_nosie_size = 0.21/4
        elif args.arch in ['ResNet18','ResNet34','ResNet50','ResNet101']:
            min_pixel = -2.42906570435
            max_pixel = 2.75373125076        
            if args.in_dataset == 'cifar10':
                random_nosie_size = 0.25/4
            elif args.in_dataset == 'cifar100':
                random_nosie_size = 0.25/8
            else:
                random_nosie_size = 0.25/4
        
        if args.in_dataset == 'cifar10':
            test_TF = cifar10_test_transform
            test_dataset = datasets.CIFAR10(root = '/home/esoc/repo/datasets/pytorch/cifar10', train=False, transform = test_TF, download=False)
        elif args.in_dataset == 'cifar100':
            test_TF = cifar100_test_transform
            test_dataset = datasets.CIFAR100(root = '/home/esoc/repo/datasets/pytorch/cifar100', train=False, transform = test_TF, download=False)
        elif args.in_dataset =='svhn':
            test_TF = svhn_test_transform
            test_dataset = datasets.SVHN(root = '/home/esoc/repo/datasets/pytorch/svhn', split = 'test', transform = test_TF, download=True)
        test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 8)
        
        print('Attack: FGSM,     Dist: ' + args.in_dataset + '\n')
        adv_data_tot, clean_data_tot, noisy_data_tot = 0, 0, 0
        label_tot = 0
        adv_noise = 0.05
        correct, adv_correct, noise_correct = 0, 0, 0
        total, generated_noise = 0, 0

        criterion = nn.CrossEntropyLoss().cuda()

        selected_list = []
        selected_index = 0
    
        for data, target in test_dataloader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.data).cpu()
            correct += equal_flag.sum()

            noisy_data = torch.add(data.data, random_nosie_size, torch.randn(data.size()).cuda()) 
            noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

            if total == 0:
                clean_data_tot = data.clone().data.cpu()
                label_tot = target.clone().data.cpu()
                noisy_data_tot = noisy_data.clone().cpu()
            else:
                clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
                label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
                noisy_data_tot = torch.cat((noisy_data_tot, noisy_data.clone().cpu()),0)
                
            # generate adversarial
            net.zero_grad()
            inputs = data.data.requires_grad_(True)
            output = net(inputs)
            loss = criterion(output, target)
            loss.backward()

            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float()-0.5)*2
            if args.arch == 'DenseNet':
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                    gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                    gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                    gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                    gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                    gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                    gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

            adv_data = torch.add(inputs.data, adv_noise, gradient)
            adv_data = torch.clamp(adv_data, min_pixel, max_pixel)
            
            # measure the noise 
            temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
            temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
            generated_noise += torch.sum(temp_noise_max)

            if total == 0:
                flag = 1
                adv_data_tot = adv_data.clone().cpu()
            else:
                adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)
            output = net(adv_data.data.requires_grad_(True))
            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag_adv = pred.eq(target.data).cpu()
            adv_correct += equal_flag_adv.sum()
            
            output = net(noisy_data.data.requires_grad_(True))
            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag_noise = pred.eq(target.data).cpu()
            noise_correct += equal_flag_noise.sum()
            
            for i in range(data.size(0)):
                if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                    selected_list.append(selected_index)
                selected_index += 1
                
            total += data.size(0)

        selected_list = torch.LongTensor(selected_list)
        clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
        adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
        noisy_data_tot = torch.index_select(noisy_data_tot, 0, selected_list)
        label_tot = torch.index_select(label_tot, 0, selected_list)

        torch.save(clean_data_tot, '%sclean_data_%s_%s_%s.pth' % (args.outf, args.arch, args.in_dataset, 'Adversarial'))
        torch.save(adv_data_tot, '%sadv_data_%s_%s_%s.pth' % (args.outf, args.arch, args.in_dataset, 'Adversarial'))
        torch.save(noisy_data_tot, '%snoisy_data_%s_%s_%s.pth' % (args.outf, args.arch, args.in_dataset, 'Adversarial'))
        torch.save(label_tot, '%slabel_%s_%s_%s.pth' % (args.outf, args.arch, args.in_dataset, 'Adversarial'))

        print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
        print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
        print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
        print('Noisy Accuracy: {}/{} ({:.2f}%)\n'.format(noise_correct, total, 100. * noise_correct / total))

    test_clean_data = torch.load(clean_data_filename)
    test_adv_data = torch.load(adv_data_filename)
    test_noisy_data = torch.load(noisy_data_filename)
    test_label = torch.load(label_filename)
    if args.tuning:
        adv_dataset = Adversarial_Dataset(test_adv_data, test_label)
        test_indices = list(range(len(adv_dataset)))
        adv_val_dataset, adv_test_dataset = Subset(adv_dataset, test_indices[:1000]), Subset(adv_dataset, test_indices[1000:])
        adv_val_dataloader = DataLoader(adv_val_dataset,batch_size=100, shuffle=True)
        adv_test_dataloader = DataLoader(adv_test_dataset,batch_size=100, shuffle=True)
        return 1, adv_val_dataloader, adv_test_dataloader
    return test_clean_data, test_adv_data, test_noisy_data, test_label

class Adversarial_Dataset(Dataset):
    def __init__(self, adv_data, adv_label):
        self.x_data = adv_data
        self.y_data = adv_label

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self,idx):
        return self.x_data[idx],self.y_data[idx]