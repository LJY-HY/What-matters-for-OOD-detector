import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.gram_detector import *


global gram_feats
gram_feats = []
global collecting 
collecting = False

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, act_func, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.act_func = act_func
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        if collecting:
            gram_feats.append(out)
        out = self.act_func(self.bn1(out))
        if collecting:
            gram_feats.append(out)
        out = self.conv2(out)
        if collecting:
            gram_feats.append(out)
        out = self.bn2(out)
        if collecting:
            gram_feats.append(out)
        out += self.shortcut(x)
        if collecting:
            gram_feats.append(self.shortcut(x))
        out = self.act_func(out)
        if collecting:
            gram_feats.append(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, act_func, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        self.act_func = act_func
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        if collecting:
            gram_feats.append(out)
        out = self.act_func(self.bn1(out))
        if collecting:
            gram_feats.append(out)
        out = self.conv2(out)
        if collecting:
            gram_feats.append(out)
        out = self.act_func(self.bn2(out))
        if collecting:
            gram_feats.append(out)
        out = self.conv3(out)
        if collecting:
            gram_feats.append(out)
        out = self.bn3(out)
        if collecting:
            gram_feats.append(out)
        out += self.shortcut(x)
        if collecting:
            gram_feats.append(self.shortcut(x))
        out = self.act_func(out)
        if collecting:
            gram_feats.append(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, act_func='relu'):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if act_func == 'relu':
            self.act_func = nn.ReLU()
        elif act_func == 'gelu':
            self.act_func = nn.GELU()
        elif act_func == 'leaky_relu':
            self.act_func = nn.LeakyReLU()
        elif act_func == 'softplus':
            self.act_func = nn.Softplus()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act_func=self.act_func)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act_func=self.act_func)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act_func=self.act_func)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, act_func=self.act_func)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, act_func):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, act_func, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = self.act_func(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = self.act_func(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)               
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = self.act_func(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate

    def gram_feature_list(self,x):
        global collecting
        collecting = True
        global gram_feats
        gram_feats = []
        self.forward(x)
        collecting = False
        temp = gram_feats
        gram_feats = []
        return temp
  
    def get_min_max(self, data, power):
        mins = []
        maxs = []
        
        for i in range(0,len(data),128):
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            for L,feat_L in enumerate(feat_list):
                if L==len(mins):
                    mins.append([None]*len(power))
                    maxs.append([None]*len(power))
                
                for p,P in enumerate(power):
                    g_p = G_p(feat_L,P)
                    
                    current_min = g_p.min(dim=0,keepdim=True)[0]
                    current_max = g_p.max(dim=0,keepdim=True)[0]
                    
                    if mins[L][p] is None:
                        mins[L][p] = current_min
                        maxs[L][p] = current_max
                    else:
                        mins[L][p] = torch.min(current_min,mins[L][p])
                        maxs[L][p] = torch.max(current_max,maxs[L][p])
        
        return mins,maxs
    
    def get_deviations(self,data,power,mins,maxs):
        deviations = []
        for i in range(0,len(data),128):     
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            batch_deviations = []
            for L,feat_L in enumerate(feat_list):
                dev = 0
                for p,P in enumerate(power):
                    g_p = G_p(feat_L,P)
                    
                    dev +=  (self.act_func(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                    dev +=  (self.act_func(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
                batch_deviations.append(dev.cpu().detach().numpy())
            batch_deviations = np.concatenate(batch_deviations,axis=1)
            deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0)
        
        return deviations

def ResNet18(args):
    return ResNet(BasicBlock, [2,2,2,2], args.num_classes, args.act_func)

def ResNet34(args):
    return ResNet(BasicBlock, [3,4,6,3], args.num_classes, args.act_func)

def ResNet50(args):
    return ResNet(Bottleneck, [3,4,6,3], args.num_classes, args.act_func)

def ResNet101(args):
    return ResNet(Bottleneck, [3,4,23,3], args.num_classes, args.act_func)

def ResNet152(args):
    return ResNet(Bottleneck, [3,8,36,3], args.num_classes, args.act_func)
