import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import argparse
from utils.arguments import get_kd_arguments
from utils.utils import *
from dataset.cifar import *

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_kd_arguments()
    args.device = torch.device('cuda',args.gpu_id)

    # dataset setting
    if args.in_dataset in ['cifar10','svhn']:
        args.num_classes=10
    elif args.in_dataset in ['cifar100']:
        args.num_classes=100

    # Get Dataset
    train_dataloader, test_dataloader = globals()[args.in_dataset](args)

    # Loss definition
    KL_loss = nn.KLDivLoss(reduction='batchmean')
    CE_loss = nn.CrossEntropyLoss()

    # Get architecture
    student_net = get_architecture(args)
    teacher_net = get_architecture(args)

    teacher_path = './checkpoint/'+args.in_dataset+'/Baseline/'+args.arch+'_trial_'+args.trial
    checkpint = torch.load(teacher_path)
    teacher_net.load_state_dict(checkpint)
    teacher_net.eval()
    

    # Get optimizer, scheduler
    optimizer, scheduler = get_optim_scheduler(args,student_net)
    teacher_opt, teacher_sche = get_optim_scheduler(args,teacher_net)

    # Test Teacher network's accuracy
    print('Teacher network Accuracy')
    acc = test(args, teacher_net, test_dataloader, optimizer, scheduler, CE_loss)
    
    student_path = './checkpoint/'+args.in_dataset+'/'+args.arch+'_'+str(args.epoch)+'_'+str(args.batch_size)+'_'+args.optimizer+'_'+args.scheduler+'_'+str(args.lr)[2:]+'_distilled_trial_'+args.trial
    best_acc=0
    for epoch in range(args.epoch):
        train(args, student_net, teacher_net, train_dataloader, optimizer, scheduler, CE_loss, KL_loss, epoch)
        acc = test(args, student_net, test_dataloader, optimizer, scheduler, CE_loss, epoch)
        scheduler.step()
        if best_acc<acc:
            best_acc = acc
            if not os.path.isdir('checkpoint/'+args.in_dataset):
                os.makedirs('checkpoint/'+args.in_dataset)
            torch.save(student_net.state_dict(), student_path)

def train(args, student_net, teacher_net, train_dataloader, optimizer, scheduler, CE_loss, KL_loss, epoch):
    student_net.train()
    train_loss = 0
    p_bar = tqdm(range(train_dataloader.__len__()))
    loss_average = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(args.device), targets.to(args.device)  
        student_outputs = student_net(inputs)
        teacher_outputs = teacher_net(inputs)   
        distillation_loss = args.alpha*args.temp*args.temp*KL_loss(F.log_softmax(student_outputs/args.temp,dim=1),F.softmax(teacher_outputs/args.temp,dim=1))
        student_loss = (1-args.alpha)*CE_loss(student_outputs,targets)
        loss =distillation_loss + student_loss

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.epoch,
                    batch=batch_idx + 1,
                    iter=train_dataloader.__len__(),
                    lr=scheduler.optimizer.param_groups[0]['lr'],
                    loss = train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    return train_loss/train_dataloader.__len__()        # average train_loss

def test(args, net, test_dataloader, optimizer, scheduler, CE_loss, epoch=0):
    net.eval()
    test_loss = 0
    acc = 0
    p_bar = tqdm(range(test_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=test_dataloader.__len__(),
                    lr=scheduler.optimizer.param_groups[0]['lr'],
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc+=sum(outputs.argmax(dim=1)==targets)
    p_bar.close()
    acc = acc/test_dataloader.dataset.__len__()
    print('Accuracy :'+ '%0.4f'%acc )
    return acc


if __name__ == '__main__':
    main()

# TODO : combine model saving/loading method