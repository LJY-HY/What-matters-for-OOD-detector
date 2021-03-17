import random
import math
import faulthandler
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
from torchvision import datasets
from torch.nn.parameter import Parameter

import os
import argparse

from utils.arguments import get_Gram_detector_arguments
from utils.utils import *
from utils.gram_detector import *

from models.resnet_big import SupConResNet, LinearClassifier
from dataset.cifar import *
from dataset.svhn import *
from dataset.non_target_data import *
from dataset.strategies import *

# import sys
# sys.stdout = open('./stdout/Gram_output.txt','a')

def main():
    faulthandler.enable()
    
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_Gram_detector_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)

    args.outf = args.outf + args.arch + '_' + args.in_dataset+'/'
    os.makedirs(args.outf,exist_ok=True)

    # dataset setting
    if args.in_dataset in ['cifar10','svhn']:
        args.num_classes=10
    elif args.in_dataset in ['cifar100']:
        args.num_classes=100
    print('load in-distribution data: ', args.in_dataset)

    # in-distribution setting
    data_train, data = globals()[args.in_dataset](args, train_TF = get_transform(args.in_dataset,'test'), test_TF = get_transform(args.in_dataset,'test'))
    data_train = list(torch.utils.data.DataLoader(data_train.dataset, batch_size = 1, shuffle = False))
    data = list(torch.utils.data.DataLoader(data.dataset,batch_size = 1, shuffle = False))
    
    # model setting
    print('load model: '+args.arch)
    net = globals()[args.arch](args).to(args.device)

    if args.e_path is not None:
        net = SupConResNet(name='resnet18', num_classes=args.num_classes).to(args.device)
        classifier = LinearClassifier(name='resnet18', num_classes=args.num_classes).to(args.device)

    # optimizer/scheduler setting
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=4e-5)
       
    # model loading
    if args.path is not None:
        checkpoint = torch.load(args.path)
    elif args.e_path is not None:
        e_checkpoint = torch.load(args.e_path)['model']
        c_checkpoint = torch.load(args.c_path)['model']
    else:
        checkpoint = torch.load('./checkpoint/'+args.in_dataset+'/'+args.arch)
    if args.e_path is None:
        net.load_state_dict(checkpoint)
        net.eval()
    else:
        net.load_state_dict(e_checkpoint)
        classifier.load_state_dict(c_checkpoint)
        net.eval()
        classifier.eval()

    # List-up OOD benchmark
    if args.in_dataset == 'cifar10':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar100']
    if args.in_dataset == 'svhn':
        out_dist_list = ['cifar10', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar100']
    if args.in_dataset == 'cifar100':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar10']


    # Extract predictions for train and test data
    train_preds = []
    train_confs = []
    train_logits = []
    for idx, (inputs, targets) in enumerate(data_train):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        if args.e_path is None:
            logits = net(inputs)
        else:
            logits = classifier(net.encoder(inputs))
        confs = F.softmax(logits,dim=1).cpu().detach().numpy()
        preds = np.argmax(confs,axis=1)
        logits = logits.cpu().detach().numpy()

        train_confs.extend(np.max(confs,axis=1))
        train_preds.extend(preds)
        train_logits.extend(logits)

    test_preds = []
    test_confs = []
    test_logits = []
    for idx, (inputs, targets) in enumerate(data):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        if args.e_path is None:
            logits = net(inputs)
        else:
            logits = classifier(net.encoder(inputs))
        confs = F.softmax(logits,dim=1).cpu().detach().numpy()
        preds = np.argmax(confs,axis=1)
        logits = logits.cpu().detach().numpy()

        test_confs.extend(np.max(confs,axis=1))
        test_preds.extend(preds)
        test_logits.extend(logits)

    # Detecting OODs by identifying anomalies in correlations
    detector = Detector(args)    
    detector.compute_minmaxs(net, data_train, train_preds, POWERS=range(1,11))
    detector.compute_test_deviations(net, data,test_preds, test_confs, POWERS=range(1,11))
    for out in out_dist_list:
        print("Out-of-distribution :",out)
        args.batch_size=1
        _,out_dataloder_test = globals()[out](args)
        ood = list(out_dataloder_test)
        results = detector.compute_ood_deviations(net, ood, args, POWERS=range(1,11), classifier=classifier)

if __name__ == '__main__':
    main()