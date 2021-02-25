import torch
import numpy as np
from models.MobileNetV2 import *
from models.ResNet import *
from models.WideResNet import *
from models.DenseNet import *
from dataset.cifar import *
import torch.optim as optim

np.random.seed(0)

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
    if args.optimizer == 'Nesterov':
        optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, nesterov= True, weight_decay=args.wd)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = args.lr)
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr = args.lr)

    if args.scheduler == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(args.epoch*0.5),int(args.epoch*0.75)],gamma=0.1)
    elif args.scheduler == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch)
    # elif args.scheduler == 'CosineWarmup':
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch)

    return optimizer, scheduler

class Rotation(object):
    def __init__(self, max_range = 4):
        pass

    def __call__(self,img):
        image_dimension = img.size().__len__()
        aug_index = np.random.randint(1,4)
        img = torch.rot90(img,aug_index, (image_dimension-2,image_dimension-1))
        return img

class CutPerm(object):
    def __init__(self, max_range = 4):
        super(CutPerm, self).__init__()
        self.max_range = max_range

    def __call__(self, img):
        _, H, W = img.size()
        aug_index = np.random.randint(1,4)
        img = self._cutperm(img, aug_index)
        return img

    def _cutperm(self, inputs, aug_index):

        _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)

        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2

        if jigsaw_h == 1:
            inputs = torch.cat((inputs[:, h_mid:, :], inputs[:, 0:h_mid, :]), dim=1)
        if jigsaw_v == 1:
            inputs = torch.cat((inputs[:, :, w_mid:], inputs[:, :, 0:w_mid]), dim=2)

        return inputs