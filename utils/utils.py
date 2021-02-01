import torch
from models.MobileNetV2 import *
from models.ResNet import *
from models.WideResNet import *
from models.DenseNet import *
from dataset.cifar import *
import torch.optim as optim

def get_architecture(args):
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
    return net

def get_optim_scheduler(args,net):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
<<<<<<< HEAD
    if args.optimizer == 'Nesterov':
        optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, nesterov= True, weight_decay=args.wd)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = args.lr)
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr = args.lr)
=======
    elif args.optimizer == 'Nesterov':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = args.lr)
    elif args.optimizer == 'LARS':
        pass
>>>>>>> 5c3ac2213488b9032326d1dccd55518dd0056116

    if args.scheduler == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epoch*0.5),int(args.epoch*0.75)],gamma=0.1)
    elif args.scheduler == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch)
    # elif args.scheduler == 'CosineWarmup':
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch)

    return optimizer, scheduler