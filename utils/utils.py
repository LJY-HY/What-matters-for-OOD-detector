import torch
from models.MobileNetV2 import *
from models.ResNet import *
from models.WideResNet import *
from models.DenseNet import *
from dataset.cifar import *
import torch.optim as optim

def get_architecture(args):
    if args.arch in ['MobileNet']:
        net = globals()[args.arch](num_classes = args.num_classes).to(args.device)
    elif args.arch in ['ResNet18','ResNet34','ResNet50','ResNet101']:
        net = globals()[args.arch](num_classes = args.num_classes).to(args.device)
    elif args.arch in ['WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4']:
        net = globals()[args.arch](num_classes = args.num_classes).to(args.device)
    elif args.arch in ['DenseNet']:
        net = globals()[args.arch](num_classes = args.num_classes).to(args.device)
    elif args.arch in ['EfficientNet']:
        pass
    return net

def get_optim_scheduler(args,net):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = args.lr)
    if args.optimizer == 'LARS':
        pass

    if args.scheduler == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epoch*0.5),int(args.epoch*0.75)],gamma=0.1)
    elif args.scheduler == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch)

    return optimizer, scheduler