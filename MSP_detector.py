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
    _, out_dataloader = globals()[args.out_dataset](args)

    # model setting
    if args.arch in ['MobileNet']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['ResNet18','ResNet34','ResNet50','ResNet101']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['SupConResNet']:
        net = globals()[args.arch]().to(args.device)
        import pdb;pdb.set_trace()
    elif args.arch in ['WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['DenseNet']:
        net = globals()[args.arch](args).to(args.device)
    elif args.arch in ['EfficientNet']:
        pass
   
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
    f1 = open(score_path+"confidence_In.txt", 'w')
    f2 = open(score_path+"confidence_Out.txt", 'w')
    p_bar = tqdm(range(in_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(in_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=in_dataloader.__len__(),
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc+=sum(outputs.argmax(dim=1)==targets)
        p_bar.close()
        acc = acc/in_dataloader.dataset.__len__()
        print('Accuracy :'+ '%0.4f'%acc )
        for batch_idx,(inputs,targets) in enumerate(in_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            for maxvalues in F.softmax(outputs,dim=1).max(dim=1)[0]:
                f1.write("{}\n".format(maxvalues))

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
    print("{:31}{:>22}".format("Out-of-distribution dataset:", args.out_dataset))
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