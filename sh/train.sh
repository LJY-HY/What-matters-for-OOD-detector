
date | tee resnet18_cifar10.txt

for trial in {0..4}
do
    # for epochs in 100 200 300
    # do
    #     python3 main.py --epoch ${epochs} \
    #     --trial ${trial}
    # done

    # for bsz in 64 256 512
    # do
    #     python3 main.py --batch_size ${bsz} \
    #     --trial ${trial}
    # done

    python3 main.py --optimizer Nesterov \
    --trial ${trial}
done

# cp ./checkpoint/cifar10/* ./checkpoint/cifar10/Iter1