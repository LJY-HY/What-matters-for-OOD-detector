import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Training Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'SVHN', 'imagenet'], help = 'dataset choice')
    parser.add_argument('--arch', default = 'ResNet18', type=str, choices = ['MobileNet','DenseNet','ResNet18','ResNet34','ResNet50','ResNet101','WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4','EfficientNet'])
    parser.add_argument('--optimizer', default = 'SGD', type=str, choices = ['SGD','Nesterov','Adam','AdamW'])
    parser.add_argument('--lr','--learning-rate', default = 0.1, type=float, choices = [1.0,0.1,0.001,0.0005,0.0002,0.0001])
    parser.add_argument('--epoch', default=300, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=128, type=int, choices=[64,128,256,512])
    parser.add_argument('--dropout_rate', default=0, type=float, choices=[0,0.3,0.5,0.7])
    parser.add_argument('--scheduler', default='MultiStepLR', type=str, choices=['MultiStepLR','CosineAnnealing','CosineWarmup'])
    parser.add_argument('--wd', '--weight_decay','--wdecay', default=5e-4, type=float, choices=[5e-4,1e-2,1e-3,1e-4,1e-6])
    parser.add_argument('--warmup',action='store_true')
    parser.add_argument('--BN','--batch_normalization',action='store_false')
    parser.add_argument('--refinement', type=str, choices=['label_smoothing','mixup'])
    parser.add_argument('--act_func', default='relu', type=str, choices=['relu','gelu','leaky_relu','softplus'])
    parser.add_argument('--tuning',action='store_true')
    parser.add_argument('--trial', default = '0', type=str)
    args = parser.parse_args()
    return args

def get_MSP_detector_arguments():
    parser = argparse.ArgumentParser(description = 'Detecting OOD Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'svhn'], help = 'in_distribution dataset')
    parser.add_argument('--arch', default = 'ResNet18', type=str, choices = ['MobileNet','DenseNet','ResNet18','ResNet34','ResNet50','ResNet101','WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4','EfficientNet'])
    parser.add_argument('--act_func', default='relu', type=str, choices=['relu','gelu','leaky_relu','softplus'])
    parser.add_argument('--batch_size', default=128, type=int, choices=[64,128,256])
    parser.add_argument('--tuning',action='store_true')
    parser.add_argument('--outf', default='./workspace/output/', help='folder to output results')
    parser.add_argument('--path', default = None, type = str, help = 'path of model to be tested')
    args = parser.parse_args()
    return args

def get_ODIN_detector_arguments():
    parser = argparse.ArgumentParser(description = 'Detecting OOD Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'svhn'], help = 'in_distribution dataset')
    parser.add_argument('--arch', default = 'ResNet18', type=str, choices = ['MobileNet','DenseNet','ResNet18','ResNet34','ResNet50','ResNet101','WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4','EfficientNet'])
    parser.add_argument('--act_func', default='relu', type=str, choices=['relu','gelu','leaky_relu','softplus'])
    parser.add_argument('--batch_size', default=128, type=int, choices=[64,128,256])
    parser.add_argument('--tuning_strategy', default='Original', type=str, choices=['Original', 'Adversarial', 'G-Odin', 'Aug_Rot', 'Aug_Perm'])
    parser.add_argument('--T', type=float)
    parser.add_argument('--ep', type=float)
    parser.add_argument('--outf', default='./workspace/output/', help='folder to output results')
    parser.add_argument('--tuning',action='store_false')
    parser.add_argument('--e_path', default = None, type = str, help='path to supcon encoder')
    parser.add_argument('--c_path', default = None, type = str, help='path to supcon classifier')
    parser.add_argument('--path', default = None, type = str, help = 'path of model to be tested')
    args = parser.parse_args()
    return args

def get_Mahalanobis_detector_arguments():
    parser = argparse.ArgumentParser(description = 'Detecting OOD Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'svhn'], help = 'in_distribution dataset')
    parser.add_argument('--arch', default = 'ResNet18', type=str, choices = ['MobileNet','DenseNet','ResNet18','ResNet34','ResNet50','ResNet101','WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4','EfficientNet'])
    parser.add_argument('--act_func', default='relu', type=str, choices=['relu','gelu','leaky_relu','softplus'])
    parser.add_argument('--batch_size', default=128, type=int, choices=[64,128,256])
    parser.add_argument('--tuning',action='store_true')
    parser.add_argument('--from_supcon',action='store_true')
    parser.add_argument('--path', default = None, type = str, help = 'path of model to be tested')
    parser.add_argument('--e_path', default = None, type = str, help='path to supcon encoder')
    parser.add_argument('--c_path', default = None, type = str, help='path to supcon classifier')
    parser.add_argument('--outf', default='./workspace/output/', help='folder to output results')
    parser.add_argument('--tuning_strategy', default='Original', type=str, choices=['Original', 'Adversarial', 'G-Odin', 'Aug_Rot', 'Aug_Perm'])
    args = parser.parse_args()
    return args


def get_Gram_detector_arguments():
    parser = argparse.ArgumentParser(description = 'Detecting OOD Arguments')
    parser.add_argument('--gpu_id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--in_dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'svhn'], help = 'in_distribution dataset')
    parser.add_argument('--arch', default = 'ResNet18', type=str, choices = ['MobileNet','DenseNet','ResNet18','ResNet34','ResNet50','ResNet101','WideResNet28_2','WideResNet28_10','WideResNet40_2','WideResNet40_4','EfficientNet'])
    parser.add_argument('--act_func', default='relu', type=str, choices=['relu','gelu','leaky_relu','softplus'])
    parser.add_argument('--batch_size', default=1, type=int, choices=[16,32,64,128,256])
    parser.add_argument('--tuning',action='store_true')
    parser.add_argument('--path', default = None, type = str, help = 'path of model to be tested')
    parser.add_argument('--outf', default='./workspace/output/', help='folder to output results')
    args = parser.parse_args()
    return args