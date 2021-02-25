import sys
import random
import matplotlib.pyplot as plt
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter

import os
from tqdm import tqdm
import argparse

from utils.arguments import get_Gram_detector_arguments
from utils.utils import *

from models.MobileNetV2 import *
from models.ResNet import *
from models.WideResNet import *
from models.DenseNet import *
from dataset.cifar import *
from dataset.svhn import *
from dataset.non_target_data import *
from dataset.strategies import *

# import sys
# sys.stdout = open('./stdout/Gram_output.txt','a')

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_Gram_detector_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    args.outf = args.outf + args.arch + '_' + args.in_dataset+'/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)

    # dataset setting
    if args.in_dataset in ['cifar10','svhn']:
        args.num_classes=10
    elif args.in_dataset in ['cifar100']:
        args.num_classes=100
    print('load in-distribution data: ', args.in_dataset)
    in_dataloader_train, in_dataloader_test = globals()[args.in_dataset](args)    #train_dataloader is not needed
    data_train = list(torch.utils.data.DataLoader(datasets.CIFAR10('data',train=True, download = True, transform = transform_test),batch_size = args.batch_size, shuffle = False))
    data = list(torch.utils.data.DataLoader(datasets.CIFAR10('data',train=False, download = True, transform = transform_test),batch_size = args.batch_size, shuffle = False))

    # model setting
    print('load model: '+args.arch)
    if args.arch in ['MobileNet']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['ResNet18','ResNet34','ResNet50','ResNet101']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['DenseNet']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['EfficientNet']:
        pass
    
    # optimizer/scheduler setting
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=4e-5)
       
    # model loading
    if args.path is not None:
        checkpoint = torch.load(args.path)
    else:
        checkpoint = torch.load('./checkpoint/'+args.in_dataset+'/'+args.arch)
    net.load_state_dict(checkpoint)
    net.eval()

    # List-up OOD benchmark
    if args.in_dataset == 'cifar10':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar100']
    if args.in_dataset == 'svhn':
        out_dist_list = ['cifar10', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar100']
    if args.in_dataset == 'cifar100':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar10']

    correct = 0
    total = 0
    for x,y in test_dataloader:
        x = x.cuda()
        y = y.numpy()
        correct +=(y==np.argmax(net(x).detach().cpu().numpy(),axis=1)).sum()
        total += y.shape[0]
    print("Accuracy :",correct/total)

    for out in out_dist_list:
        _,out_dataloder_test = globals()[out](args)