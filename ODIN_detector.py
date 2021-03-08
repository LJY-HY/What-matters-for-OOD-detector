from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import argparse

from utils.arguments import get_ODIN_detector_arguments
from utils import calMetric
from utils.utils import *

from models.resnet_big import SupConResNet, LinearClassifier
from dataset.cifar import *
from dataset.svhn import *
from dataset.non_target_data import *
from dataset.strategies import *

# import sys
# sys.stdout = open('./stdout/ODIN_output.txt','a')

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_ODIN_detector_arguments()
    args.device = torch.device('cuda',args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # dataset setting
    if args.in_dataset in ['cifar10','svhn']:
        args.num_classes=10
    elif args.in_dataset in ['cifar100']:
        args.num_classes=100
    _, in_dataloader_val, in_dataloader_test = globals()[args.in_dataset](args)    # train dataset is not needed
   
    args.outf = args.outf + args.arch + '_' + args.in_dataset+'/'

    # model setting/ loading
    net = get_architecture(args)
   
    if args.e_path is not None:
        net = SupConResNet(name='resnet18', num_classes=args.num_classes).to(args.device)
        classifier = LinearClassifier(name='resnet18', num_classes=args.num_classes).to(args.device)

    if args.in_dataset == 'cifar10':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar100']
    if args.in_dataset == 'svhn':
        out_dist_list = ['cifar10', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar100']
    if args.in_dataset == 'cifar100':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX','cifar10']

    # optimizer/ scheduler setting
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

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
    criterion = nn.CrossEntropyLoss()

    # Softmax Scores Path Setting
    score_path = './workspace/softmax_scores/'
    os.makedirs(score_path,exist_ok=True)

    # Adversarial samples clearing
    adversarial_data_path = './output/'+args.arch+'_'+args.in_dataset+'/adv_data_'+args.arch+'_'+args.in_dataset+'_Adversarial.pth'
    clean_data_path = './output/'+args.arch+'_'+args.in_dataset+'/clean_data_'+args.arch+'_'+args.in_dataset+'_Adversarial.pth'
    if os.path.isfile(adversarial_data_path):
        os.remove(adversarial_data_path)
    if os.path.isfile(clean_data_path):
        os.remove(clean_data_path)

    if args.tuning_strategy is not 'Original':
        print('Tuning with strategy\n')
        _, out_dataloader_val, _ = globals()[args.tuning_strategy](args)
        tnr_best=0.
        T_temp=1
        ep_temp=0
        T_candidate = [1,10,100,1000]
        e_candidate = [0,0.0005,0.001,0.0014,0.002,0.0024,0.005,0.01,0.05,0.1,0.2]
        # Tuning
        for T in T_candidate:
            for ep in e_candidate:
                ##### Open files to save confidence score #####
                f1 = open(score_path+"confidence_Base_In.txt", 'w')
                f2 = open(score_path+"confidence_Base_Out.txt", 'w')
                g1 = open(score_path+"confidence_In.txt", 'w')
                g2 = open(score_path+"confidence_Out.txt", 'w')
                # processing in-distribution data
                for batch_idx, data in enumerate(in_dataloader_val):
                    images, _ = data
                    inputs = Variable(images.to(args.device),requires_grad=True)
                    del images
                    if args.e_path is None:
                        outputs = net(inputs)
                    else:
                        outputs = classifier(net.encoder(inputs))
                    if len(outputs)==2:
                        outputs = outputs[0]
                    nnOutputs = outputs.data.cpu()
                    nnOutputs = nnOutputs.numpy()
                    nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
                    nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
                    nnOutputs = nnOutputs.transpose()
                    for maxvalues in np.max(nnOutputs,axis=1):
                        f1.write("{}\n".format(maxvalues,axis=1))

                    outputs = outputs / T
                    maxIndexTemp = np.argmax(nnOutputs,axis=1)
                    labels = Variable(torch.LongTensor([maxIndexTemp]).to(args.device))
                    loss = criterion(outputs, labels[0])
                    loss.backward()
                    
                    # Normalizing the gradient to binary in {0, 1}
                    gradient =  torch.ge(inputs.grad.data, 0)
                    gradient = (gradient.float() - 0.5) * 2
                    # Normalizing the gradient to the same space of image
                    gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
                    gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
                    gradient[0][2] = (gradient[0][2])/(66.7/255.0)
                    # Adding small perturbations to images
                    tempInputs = torch.add(inputs.data, gradient, alpha=-ep)

                    # Now re-input noise-added input(tempInputs)
                    if args.e_path is None:
                        outputs = net(Variable(tempInputs))
                    else:
                        outputs = classifier(net.encoder(Variable(tempInputs)))
                    if len(outputs)==2:
                        outputs = outputs[0]
                    outputs = outputs / T
                    # Calculating the confidence after adding perturbations
                    nnOutputs = outputs.data.cpu()
                    nnOutputs = nnOutputs.numpy()
                    nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
                    nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
                    nnOutputs = nnOutputs.transpose()
                    for maxvalues in np.max(nnOutputs,axis=1):
                        g1.write("{}\n".format(maxvalues))
                   
                f1.close()
                g1.close()
                # Processing out-of-distribution data
                for batch_idx, data in enumerate(out_dataloader_val):
                    images, _ = data
                    inputs = Variable(images.to(args.device),requires_grad=True)
                    del images
                    if args.e_path is None:
                        outputs = net(inputs)
                    else:
                        outputs = classifier(net.encoder(inputs))
                    if len(outputs)==2:
                        outputs = outputs[0]
                    nnOutputs = outputs.data.cpu()
                    nnOutputs = nnOutputs.numpy()
                    nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
                    nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
                    nnOutputs = nnOutputs.transpose()
                    for maxvalues in np.max(nnOutputs,axis=1):
                        f2.write("{}\n".format(maxvalues,axis=1))

                    outputs = outputs / T
                    maxIndexTemp = np.argmax(nnOutputs,axis=1)
                    labels = Variable(torch.LongTensor([maxIndexTemp]).to(args.device))
                    loss = criterion(outputs, labels[0])
                    loss.backward()
                    
                    # Normalizing the gradient to binary in {0, 1}
                    gradient =  torch.ge(inputs.grad.data, 0)
                    gradient = (gradient.float() - 0.5) * 2
                    # Normalizing the gradient to the same space of image
                    gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
                    gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
                    gradient[0][2] = (gradient[0][2])/(66.7/255.0)
                    # Adding small perturbations to images
                    tempInputs = torch.add(inputs.data, gradient, alpha=-ep)

                    # Now re-input noise-added input(tempInputs)
                    if args.e_path is None:
                        outputs = net(Variable(tempInputs))
                    else:
                        outputs = classifier(net.encoder(Variable(tempInputs)))
                    if len(outputs)==2:
                        outputs = outputs[0]
                    outputs = outputs / T
                    # Calculating the confidence after adding perturbations
                    nnOutputs = outputs.data.cpu()
                    nnOutputs = nnOutputs.numpy()
                    nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
                    nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
                    nnOutputs = nnOutputs.transpose()
                    for maxvalues in np.max(nnOutputs,axis=1):
                        g2.write("{}\n".format(maxvalues))
                f2.close()
                g2.close()
                # calculate metrics
                result = calMetric.metric(score_path)
                if tnr_best<result['TNR']:
                    tnr_best=result['TNR']
                    results_best = result
                    T_temp=T
                    ep_temp=ep
        print('Tuned T       : ',T_temp)
        print('Tuned epsilon : ',ep_temp)
        T = T_temp
        ep = ep_temp

    for out in out_dist_list:
        _, out_dataloader_val, out_dataloader_test = globals()[out](args)
        if args.tuning_strategy == 'Original':
            tnr_best=0.
            T_temp=1
            ep_temp=0
            print('Tuning with no strategy\n')
            T_candidate = [1,10,100,1000]
            e_candidate = [0,0.0005,0.001,0.0014,0.002,0.0024,0.005,0.01,0.05,0.1,0.2]
            # Tuning
            for T in T_candidate:
                for ep in e_candidate:
                    ##### Open files to save confidence score #####
                    f1 = open(score_path+"confidence_Base_In.txt", 'w')
                    f2 = open(score_path+"confidence_Base_Out.txt", 'w')
                    g1 = open(score_path+"confidence_In.txt", 'w')
                    g2 = open(score_path+"confidence_Out.txt", 'w')
                    # processing in-distribution data
                    for batch_idx, data in enumerate(in_dataloader_val):
                        images, _ = data
                        inputs = Variable(images.to(args.device),requires_grad=True)
                        del images
                        if args.e_path is None:
                            outputs = net(inputs)
                        else:
                            outputs = classifier(net.encoder(inputs))
                        if len(outputs)==2:
                            outputs = outputs[0]
                        nnOutputs = outputs.data.cpu()
                        nnOutputs = nnOutputs.numpy()
                        nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
                        nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
                        nnOutputs = nnOutputs.transpose()
                        for maxvalues in np.max(nnOutputs,axis=1):
                            f1.write("{}\n".format(maxvalues,axis=1))

                        outputs = outputs / T
                        maxIndexTemp = np.argmax(nnOutputs,axis=1)
                        labels = Variable(torch.LongTensor([maxIndexTemp]).to(args.device))
                        loss = criterion(outputs, labels[0])
                        loss.backward()
                        
                        # Normalizing the gradient to binary in {0, 1}
                        gradient =  torch.ge(inputs.grad.data, 0)
                        gradient = (gradient.float() - 0.5) * 2
                        # Normalizing the gradient to the same space of image
                        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
                        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
                        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
                        # Adding small perturbations to images
                        tempInputs = torch.add(inputs.data, gradient, alpha=-ep)

                        # Now re-input noise-added input(tempInputs)
                        if args.e_path is None:
                            outputs = net(Variable(tempInputs))
                        else:
                            outputs = classifier(net.encoder(Variable(tempInputs)))
                        if len(outputs)==2:
                            outputs = outputs[0]
                        outputs = outputs / T
                        # Calculating the confidence after adding perturbations
                        nnOutputs = outputs.data.cpu()
                        nnOutputs = nnOutputs.numpy()
                        nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
                        nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
                        nnOutputs = nnOutputs.transpose()
                        for maxvalues in np.max(nnOutputs,axis=1):
                            g1.write("{}\n".format(maxvalues))
                    
                    f1.close()
                    g1.close()
                    # Processing out-of-distribution data
                    for batch_idx, data in enumerate(out_dataloader_val):
                        images, _ = data
                        inputs = Variable(images.to(args.device),requires_grad=True)
                        del images
                        if args.e_path is None:
                            outputs = net(inputs)
                        else:
                            outputs = classifier(net.encoder(inputs))
                        if len(outputs)==2:
                            outputs = outputs[0]
                        nnOutputs = outputs.data.cpu()
                        nnOutputs = nnOutputs.numpy()
                        nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
                        nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
                        nnOutputs = nnOutputs.transpose()
                        for maxvalues in np.max(nnOutputs,axis=1):
                            f2.write("{}\n".format(maxvalues,axis=1))

                        outputs = outputs / T
                        maxIndexTemp = np.argmax(nnOutputs,axis=1)
                        labels = Variable(torch.LongTensor([maxIndexTemp]).to(args.device))
                        loss = criterion(outputs, labels[0])
                        loss.backward()
                        
                        # Normalizing the gradient to binary in {0, 1}
                        gradient =  torch.ge(inputs.grad.data, 0)
                        gradient = (gradient.float() - 0.5) * 2
                        # Normalizing the gradient to the same space of image
                        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
                        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
                        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
                        # Adding small perturbations to images
                        tempInputs = torch.add(inputs.data, gradient, alpha=-ep)

                        # Now re-input noise-added input(tempInputs)
                        if args.e_path is None:
                            outputs = net(Variable(tempInputs))
                        else:
                            outputs = classifier(net.encoder(Variable(tempInputs)))
                        if len(outputs)==2:
                            outputs = outputs[0]
                        outputs = outputs / T
                        # Calculating the confidence after adding perturbations
                        nnOutputs = outputs.data.cpu()
                        nnOutputs = nnOutputs.numpy()
                        nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
                        nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
                        nnOutputs = nnOutputs.transpose()
                        for maxvalues in np.max(nnOutputs,axis=1):
                            g2.write("{}\n".format(maxvalues))
                    f2.close()
                    g2.close()
                    # calculate metrics
                    result = calMetric.metric(score_path)
                    if tnr_best<result['TNR']:
                        tnr_best=result['TNR']
                        results_best = result
                        T_temp=T
                        ep_temp=ep
            print('Tuned T       : ',T_temp)
            print('Tuned epsilon : ',ep_temp)
            T = T_temp
            ep = ep_temp

        ##### Open files to save confidence score #####
        f1 = open(score_path+"confidence_Base_In.txt", 'w')
        f2 = open(score_path+"confidence_Base_Out.txt", 'w')
        g1 = open(score_path+"confidence_In.txt", 'w')
        g2 = open(score_path+"confidence_Out.txt", 'w')
        # processing with tuned T,epsilon
        for batch_idx, data in enumerate(in_dataloader_test):
            images, _ = data
            inputs = Variable(images.to(args.device),requires_grad=True)
            del images
            if args.e_path is None:
                outputs = net(inputs)
            else:
                outputs = classifier(net.encoder(inputs))
            if len(outputs)==2:
                outputs = outputs[0]
            nnOutputs = outputs.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
            nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
            nnOutputs = nnOutputs.transpose()
            for maxvalues in np.max(nnOutputs,axis=1):
                f1.write("{}\n".format(maxvalues,axis=1))

            outputs = outputs / T
            maxIndexTemp = np.argmax(nnOutputs,axis=1)
            labels = Variable(torch.LongTensor([maxIndexTemp]).to(args.device))
            loss = criterion(outputs, labels[0])
            loss.backward()
            
            # Normalizing the gradient to binary in {0, 1}
            gradient =  torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[0][0] = (gradient[0][0])/(63.0/255.0)
            gradient[0][1] = (gradient[0][1])/(62.1/255.0)
            gradient[0][2] = (gradient[0][2])/(66.7/255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(inputs.data, gradient, alpha=-ep)

            # Now re-input noise-added input(tempInputs)
            if args.e_path is None:
                outputs = net(Variable(tempInputs))
            else:
                outputs = classifier(net.encoder(Variable(tempInputs)))
            if len(outputs)==2:
                outputs = outputs[0]
            outputs = outputs / T
            # Calculating the confidence after adding perturbations
            nnOutputs = outputs.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
            nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
            nnOutputs = nnOutputs.transpose()
            for maxvalues in np.max(nnOutputs,axis=1):
                g1.write("{}\n".format(maxvalues))
        f1.close()
        g1.close()
        # Processing out-of-distribution data
        for batch_idx, data in enumerate(out_dataloader_test):
            images, _ = data
            inputs = Variable(images.to(args.device),requires_grad=True)
            del images
            if args.e_path is None:
                outputs = net(inputs)
            else:
                outputs = classifier(net.encoder(inputs))
            if len(outputs)==2:
                outputs = outputs[0]
            nnOutputs = outputs.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
            nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
            nnOutputs = nnOutputs.transpose()
            for maxvalues in np.max(nnOutputs,axis=1):
                f2.write("{}\n".format(maxvalues,axis=1))

            outputs = outputs / T
            maxIndexTemp = np.argmax(nnOutputs,axis=1)
            labels = Variable(torch.LongTensor([maxIndexTemp]).to(args.device))
            loss = criterion(outputs, labels[0])
            loss.backward()
            
            # Normalizing the gradient to binary in {0, 1}
            gradient =  torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
            gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
            gradient[0][2] = (gradient[0][2])/(66.7/255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(inputs.data, gradient, alpha=-ep)

            # Now re-input noise-added input(tempInputs)
            if args.e_path is None:
                outputs = net(Variable(tempInputs))
            else:
                outputs = classifier(net.encoder(Variable(tempInputs)))
            if len(outputs)==2:
                outputs = outputs[0]
            outputs = outputs / T
            # Calculating the confidence after adding perturbations
            nnOutputs = outputs.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = np.subtract(nnOutputs.transpose(),np.max(nnOutputs,axis=1)).transpose()
            nnOutputs = np.exp(nnOutputs).transpose()/np.sum(np.exp(nnOutputs),axis=1)
            nnOutputs = nnOutputs.transpose()
            for maxvalues in np.max(nnOutputs,axis=1):
                g2.write("{}\n".format(maxvalues))
        f2.close()
        g2.close()
        # calculate metrics
        result = calMetric.metric(score_path)
        mtypes = ['TNR', 'DTACC', 'AUROC', 'AUIN', 'AUOUT']
        print('\nBest Performance Out-of-Distribution Detection')
        print('T       : ', T)
        print('epsilon : ', ep)
        print("{:31}{:>22}".format("Neural network architecture:", args.arch))
        print("{:31}{:>22}".format("Tuning Strategy:", args.tuning_strategy))
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