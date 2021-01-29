
date | tee resnet18_cifar10.txt

for trial in {0..4}
do
    python3 main.py --optimizer Adam \
    --trial ${trial}
done

# cp ./checkpoint/cifar10/* ./checkpoint/cifar10/Iter1