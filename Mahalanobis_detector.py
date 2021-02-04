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

from models.MobileNetV2 import *
from models.ResNet import *
from models.WideResNet import *
from models.DenseNet import *
from dataset.cifar import *
from dataset.svhn import *
from dataset.non_target_data import *

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# np.random.seed(1)

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_Mahalanobis_detector_arguments()
    args.device = torch.device('cuda',args.gpu_id)

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
    # 이건 SGD이던 Adam이던 상관없음. 어차피 net의 weight/bias는 바꾸지 않을꺼고 
    # input-preprocessing도 sign값만 가지고 하기 때문에
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

    # set information about feature extaction
    temp_x = torch.rand(2,3,32,32).cuda()
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
    out_dist_list = ['svhn', 'LSUN', 'TinyImagenet','cifar100']

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
            file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.in_dataset , out_dist))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(file_name, Mahalanobis_data)
   
    dataset_list = [args.in_dataset]
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']

    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        print('In-distribution: ', dataset)
        outf = './output/' + args.arch + '_' + dataset + '/'
        out_list = ['svhn', 'LSUN', 'TinyImagenet', 'cifar100']
        if dataset == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']

        list_best_results_out, list_best_results_index_out = [], []
        for out in out_list:
            print('Out-of-distribution: ', out)
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, out)
                X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
                Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
                X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
                Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
                lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
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
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        out_list = ['svhn', 'LSUN', 'TinyImagenet','cifar100']
        if dataset_list[count_in] == 'svhn':
            out_list = ['cifar10', 'LSUN', 'TinyImagenet','cifar100']
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

# TODO : 잘 되는지 확인할것