import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import argparse

from utils.arguments import get_MSP_detector_arguments
from utils import calMetric

from models.MobileNetV2 import *
from models.ResNet import *
from models.resnet_big import SupConResNet
from models.WideResNet import *
from models.DenseNet import *
from dataset.cifar import *
from dataset.svhn import *
from dataset.non_target_data import *

# import sys
# sys.stdout = open('./stdout/MSP_output.txt','a')

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_MSP_detector_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    
    # dataset setting
    if args.in_dataset in ['cifar10','svhn']:
        args.num_classes=10
    elif args.in_dataset in ['cifar100']:
        args.num_classes=100

    _, in_dataloader = globals()[args.in_dataset](args)    #train_dataloader is not needed

    # model setting
    net = get_architecture(args)
      
    # optimizer/scheduler setting
    # 이건 SGD이던 Adam이던 상관없음
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=4e-5)

    # model loading
    if args.path is not None:
        checkpoint = torch.load(args.path)
    else:
        checkpoint = torch.load('./checkpoint/'+args.in_dataset+'/'+args.arch)
    net.load_state_dict(checkpoint)
    net.eval()

    # Softmax Scores Path Setting
    score_path = './workspace/softmax_scores/'
    os.makedirs(score_path,exist_ok=True)

    test_loss = 0
    acc = 0
    if args.in_dataset == 'cifar10':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar100']
    if args.in_dataset == 'svhn':
        out_dist_list = ['cifar10', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar100']
    if args.in_dataset == 'cifar100':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar10']
    for out in out_dist_list:
        _, out_dataloader = globals()[out](args)
        f1 = open(score_path+"confidence_In.txt", 'w')
        f2 = open(score_path+"confidence_Out.txt", 'w')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(in_dataloader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = net(inputs)
                loss = F.cross_entropy(outputs, targets)
                test_loss += loss.item()
                acc+=sum(outputs.argmax(dim=1)==targets)
            acc = acc/in_dataloader.dataset.__len__()
            print('Accuracy :'+ '%0.4f'%acc )
            for batch_idx,(inputs,targets) in enumerate(in_dataloader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = net(inputs)
                for maxvalues in F.softmax(outputs,dim=1).max(dim=1)[0]:
                    f1.write("{}\n".format(maxvalues))
            print('Out-of-distribution :'+out)
            for batch_idx,(inputs,targets) in enumerate(out_dataloader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = net(inputs)
                for maxvalues in F.softmax(outputs,dim=1).max(dim=1)[0]:
                    f2.write("{}\n".format(maxvalues))
        f1.close()
        f2.close()

        result = calMetric.metric(score_path)
        mtypes = ['TNR', 'DTACC', 'AUROC', 'AUIN', 'AUOUT']

        print('\nBest Performance Out-of-Distribution Detection')
        print("{:31}{:>22}".format("Neural network architecture:", args.arch))
        print("{:31}{:>22}".format("In-distribution dataset:", args.in_dataset))
        print("{:31}{:>22}".format("Out-of-distribution dataset:", out))
        print("")
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*result['TNR']), end='')
        print(' {val:6.2f}'.format(val=100.*result['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*result['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*result['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*result['AUOUT']), end='')

if __name__ == '__main__':
    main()