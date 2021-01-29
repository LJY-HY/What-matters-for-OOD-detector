import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Knowledge Distillation OOD-detection')
    parser_temp = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'SVHN', 'imagenet'], help = 'dataset choice')
    parser.add_argument('--arch', default = 'ResNet18', type=str, choices = ['MobileNet','DenseNet','ResNet18','ResNet34','ResNet50','ResNet101','WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4','EfficientNet'])
    parser.add_argument('--optimizer', default = 'SGD', type=str, choices = ['SGD','Adam'])
    parser.add_argument('--lr','--learning-rate', default = 0.1, type=float, choices = [1.0,0.1,0.001,0.0005,0.0002])
    parser.add_argument('--epoch', default=200, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=128, type=int, choices=[64,128,256,512])
    parser.add_argument('--dropout_rate', default=0, type=float, choices=[0,0.3,0.5,0.7])
    parser.add_argument('--scheduler', default='MultiStepLR', type=str, choices=['MultiStepLR','CosineAnnealing'])
    parser.add_argument('--wd', '--weight_decay','--wdecay', default=5e-4, type=float, choices=[5e-4,1e-4,1e-6])
    parser.add_argument('--warmup',action='store_true')
    parser.add_argument('--BN','--batch_normalization',action='store_false')
    parser.add_argument('--tuning',action='store_false')
    parser.add_argument('--nesterov',action='store_false')
    parser.add_argument('--trial', default = '0', type=str)
    args = parser.parse_args()
    return args