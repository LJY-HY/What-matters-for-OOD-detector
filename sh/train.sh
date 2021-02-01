python3 main.py --wd 1e-3 --gpu_id 1 --trial 0
cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_0 ./checkpoint/cifar10/weight_decay/ResNet18_1e_3_iter1
python3 main.py --wd 1e-3 --gpu_id 1 --trial 1
cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_1 ./checkpoint/cifar10/weight_decay/ResNet18_1e_3_iter2
# python3 main.py --wd 1e-3 --gpu_id 1 --trial 2
# cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_2 ./checkpoint/cifar10/weight_decay/ResNet18_1e_3_iter3
# python3 main.py --wd 1e-3 --gpu_id 1 --trial 3
# cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_3 ./checkpoint/cifar10/weight_decay/ResNet18_1e_3_iter4
# python3 main.py --wd 1e-3 --gpu_id 1 --trial 4
# cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_4 ./checkpoint/cifar10/weight_decay/ResNet18_1e_3_iter5

python3 main.py --wd 1e-2 --gpu_id 1 --trial 0
cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_0 ./checkpoint/cifar10/weight_decay/ResNet18_1e_2_iter1
python3 main.py --wd 1e-2 --gpu_id 1 --trial 1
cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_1 ./checkpoint/cifar10/weight_decay/ResNet18_1e_2_iter2
python3 main.py --wd 1e-2 --gpu_id 1 --trial 2
cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_2 ./checkpoint/cifar10/weight_decay/ResNet18_1e_2_iter3
python3 main.py --wd 1e-2 --gpu_id 1 --trial 3
cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_3 ./checkpoint/cifar10/weight_decay/ResNet18_1e_2_iter4
python3 main.py --wd 1e-2 --gpu_id 1 --trial 4
cp ./checkpoint/cifar10/ResNet18_128_SGD_MultiStepLR_1trial_4 ./checkpoint/cifar10/weight_decay/ResNet18_1e_2_iter5

python3 main.py --optimizer Adam --lr 0.001 --gpu_id 1
cp ./checkpoint/cifar10/ResNet18_128_Adam_MultiStepLR_001trial_0 ./checkpoint/cifar10/optimizer/ResNet18_Adam_iter1
python3 main.py --optimizer Adam --lr 0.001 --gpu_id 1
cp ./checkpoint/cifar10/ResNet18_128_Adam_MultiStepLR_001trial_1 ./checkpoint/cifar10/optimizer/ResNet18_Adam_iter2
python3 main.py --optimizer Adam --lr 0.001 --gpu_id 1
cp ./checkpoint/cifar10/ResNet18_128_Adam_MultiStepLR_001trial_2 ./checkpoint/cifar10/optimizer/ResNet18_Adam_iter3
python3 main.py --optimizer Adam --lr 0.001 --gpu_id 1
cp ./checkpoint/cifar10/ResNet18_128_Adam_MultiStepLR_001trial_3 ./checkpoint/cifar10/optimizer/ResNet18_Adam_iter4
python3 main.py --optimizer Adam --lr 0.001 --gpu_id 1
cp ./checkpoint/cifar10/ResNet18_128_Adam_MultiStepLR_001trial_4 ./checkpoint/cifar10/optimizer/ResNet18_Adam_iter5

