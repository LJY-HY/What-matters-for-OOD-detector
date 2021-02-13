import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegressionCV
from scipy import misc
from torch.autograd import Variable

import os
from tqdm import tqdm
import argparse

from utils.arguments import get_Mahalanobis_detector_arguments
from utils import lib_generation,lib_regression
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
# sys.stdout = open('./stdout/Mahalanobis_output.txt','a')

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_Mahalanobis_detector_arguments()
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

    # set information about feature extaction
    temp_x = torch.rand(2,3,32,32).to(args.device)
    temp_x = Variable(temp_x)
    temp_list = net.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(net, args.num_classes, feature_list, in_dataloader_train)
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    if args.in_dataset == 'cifar10':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX', 'cifar100']
    elif args.in_dataset == 'svhn':
        out_dist_list = ['cifar10', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX', 'cifar100']
    elif args.in_dataset == 'cifar100':
        out_dist_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet','TinyImagenet_FIX', 'cifar10']

    # if tuning strategy exists, find best magnitude
    if args.tuning_strategy is not 'Original':
        for magnitude in m_list:
            print('Noise: '+str(magnitude))
            if args.tuning_strategy in ['Aug_Rot','Aug_Perm','G-Odin']:
                for i in range(num_output):
                    M_in = lib_generation.get_Mahalanobis_score(net, in_dataloader_test, args.num_classes, args.outf, \
                                                                True, args.arch, sample_mean, precision, i, magnitude)
                    M_in = np.asarray(M_in, dtype=np.float32)
                    if i == 0:
                        Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                    else:
                        Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            
                _, out_test_loader = globals()[args.tuning_strategy](args)
                print('Out-distribution: ' + args.tuning_strategy) 
                for i in range(num_output):
                    M_out = lib_generation.get_Mahalanobis_score(net, out_test_loader, args.num_classes, args.outf, \
                                                                    False, args.arch, sample_mean, precision, i, magnitude)
                    M_out = np.asarray(M_out, dtype=np.float32)
                    if i == 0:
                        Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                    else:
                        Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

                Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
                Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
                Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
                file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.in_dataset , args.tuning_strategy))
                Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
                np.save(file_name, Mahalanobis_data)

            elif args.tuning_strategy in ['Adversarial']:
                test_clean_data, test_adv_data, test_noisy_data, test_label = globals()[args.tuning_strategy](args)
                for i in range(num_output):
                    M_in = lib_generation.get_Mahalanobis_score_adv(net, test_clean_data, test_label, args.num_classes, args.outf, args.arch, sample_mean, precision, i, magnitude)
                    M_in = np.asarray(M_in, dtype=np.float32)
                    if i == 0:
                        Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
                    else:
                        Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
                for i in range(num_output):
                    M_out = lib_generation.get_Mahalanobis_score_adv(net, test_adv_data, test_label, args.num_classes, args.outf, args.arch, sample_mean, precision, i, magnitude)
                    M_out = np.asarray(M_out, dtype=np.float32)
                    if i == 0:
                        Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                    else:
                        Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
                        
                for i in range(num_output):
                    M_noisy = lib_generation.get_Mahalanobis_score_adv(net, test_noisy_data, test_label, args.num_classes, args.outf, args.arch, sample_mean, precision, i, magnitude)
                    M_noisy = np.asarray(M_noisy, dtype=np.float32)
                    if i == 0:
                        Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
                    else:
                        Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)            
                Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
                Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
                Mahalanobis_noisy = np.asarray(Mahalanobis_noisy, dtype=np.float32)
                Mahalanobis_pos = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))

                Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_pos)
                file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.in_dataset, 'Adversarial'))
                
                Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
                np.save(file_name, Mahalanobis_data)
    
        score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']

        list_best_results, list_best_results_index = [], []
        print('In-distribution: ', args.in_dataset)
        outf = './workspace/output/' + args.arch + '_' + args.in_dataset + '/'
        list_best_results_out, list_best_results_index_out = [], []

        print('Out-of-distribution: ', args.tuning_strategy)
        best_tnr, best_result, best_index = 0, 0, 0
        for score in score_list:
            total_X, total_Y = lib_regression.load_characteristics(score, args.in_dataset, args.tuning_strategy, outf)
            if args.tuning_strategy in ['Aug_Rot','Aug_Perm','G-Odin']:
                X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, args.tuning_strategy)
                X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
                Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
                X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
                Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
            elif args.tuning_strategy in ['Adversarial']:
                X_val, Y_val, X_test, Y_test = lib_regression.block_split_adv(total_X, total_Y)
                pivot = int(X_val.shape[0] / 6)
                X_train = np.concatenate((X_val[:pivot], X_val[2*pivot:3*pivot], X_val[4*pivot:5*pivot]))
                Y_train = np.concatenate((Y_val[:pivot], Y_val[2*pivot:3*pivot], Y_val[4*pivot:5*pivot]))
                X_val_for_test = np.concatenate((X_val[pivot:2*pivot], X_val[3*pivot:4*pivot], X_val[5*pivot:]))
                Y_val_for_test = np.concatenate((Y_val[pivot:2*pivot], Y_val[3*pivot:4*pivot], Y_val[5*pivot:]))
            lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
            y_pred = lr.predict_proba(X_train)[:, 1]
            y_pred = lr.predict_proba(X_val_for_test)[:, 1]
            results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
            if best_tnr < results['TNR']:
                best_tnr = results['TNR']
                best_index = score
                best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
                best_lr = lr
                
        list_best_results_out.append(best_result)
        list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)
        # print the results
        count_in = 0
        mtypes = ['TNR', 'DTACC', 'AUROC', 'AUIN', 'AUOUT']
        for in_list in list_best_results:
            print('in_distribution: ' + args.in_dataset + '==========')
            count_out = 0
            for results in in_list:
                print('out_distribution: '+ args.tuning_strategy+'\n')
                for mtype in mtypes:
                    print(' {mtype:6s}'.format(mtype=mtype), end='')
                print('\n{val:6.2f}'.format(val=100.*results['TNR']), end='')
                print(' {val:6.2f}'.format(val=100.*results['DTACC']), end='')
                print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
                print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
                print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
                print('Input noise: ' + list_best_results_index[count_in][count_out])      
                print('')
                count_out += 1
            count_in += 1
    
    if args.tuning_strategy is not 'Original':
        m_list = [float(list_best_results_index[0][0][12:])]

    for magnitude in m_list:
        print('Noise: '+str(magnitude))
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score(net, in_dataloader_test, args.num_classes, args.outf, \
                                                        True, args.arch, sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
    
        for out_dist in out_dist_list:
            _, out_test_loader = globals()[out_dist](args)
            print('Out-distribution: ' + out_dist) 
            for i in range(num_output):
                M_out = lib_generation.get_Mahalanobis_score(net, out_test_loader, args.num_classes, args.outf, \
                                                                False, args.arch, sample_mean, precision, i, magnitude)
                M_out = np.asarray(M_out, dtype=np.float32)
                if i == 0:
                    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                else:
                    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.in_dataset , out_dist))      #in/out/magnitude의 모든 조합을 계산. 만약 strategy로 다른 데이터셋을 쓴다면?
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)
    
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    if args.tuning_strategy is not 'Original':
        score_list = list_best_results_index[0]
    list_best_results, list_best_results_index = [], []
    print('In-distribution: ', args.in_dataset)
    outf = './workspace/output/' + args.arch + '_' + args.in_dataset + '/'
    if args.in_dataset == 'cifar10':
        out_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet', 'TinyImagenet_FIX', 'cifar100']
    elif args.in_dataset == 'svhn':
        out_list = ['cifar10', 'LSUN', 'LSUN_FIX', 'TinyImagenet', 'TinyImagenet_FIX', 'cifar100']
    elif args.in_dataset == 'cifar100':
        out_list = ['svhn', 'LSUN', 'LSUN_FIX', 'TinyImagenet', 'TinyImagenet_FIX', 'cifar10']
    list_best_results_out, list_best_results_index_out = [], []

    for out in out_list:
        print('Out-of-distribution: ', out)
        best_tnr, best_result, best_index = 0, 0, 0
        for score in score_list:
            total_X, total_Y = lib_regression.load_characteristics(score, args.in_dataset, out, outf)
            X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, out)
            X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
            Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
            X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
            Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
            lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
            if args.tuning_strategy is not 'Original':
                lr = best_lr
            y_pred = lr.predict_proba(X_train)[:, 1]
            y_pred = lr.predict_proba(X_val_for_test)[:, 1]
            results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
            if best_tnr < results['TNR']:
                best_tnr = results['TNR']
                best_index = score
                best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            
        list_best_results_out.append(best_result)
        list_best_results_index_out.append(best_index)
    list_best_results.append(list_best_results_out)
    list_best_results_index.append(list_best_results_index_out)
        
    # print the results
    count_in = 0
    mtypes = ['TNR', 'DTACC', 'AUROC', 'AUIN', 'AUOUT']
    for in_list in list_best_results:
        print('in_distribution: ' + args.in_dataset + '==========')
        if args.in_dataset=='cifar10':
            out_list = ['svhn', 'LSUN', 'LSUN_FIX','TinyImagenet','TinyImagenet_FIX','cifar100']
        elif args.in_dataset == 'svhn':
            out_list = ['cifar10', 'LSUN', 'TinyImagenet','cifar100']
        elif args.in_dataset == 'cifar100':
            out_list = ['svhn', 'LSUN', 'LSUN_FIX','TinyImagenet','TinyImagenet_FIX','cifar10']
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ out_list[int(count_out)]+'\n')
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index[count_in][count_out])      
            print('')
            count_out += 1
        count_in += 1
if __name__ == '__main__':
    main()