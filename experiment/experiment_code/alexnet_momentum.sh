#cifar10
#python train_moment.py  -a alexnet  --dataset cifar10 --drop 0.3 --epochs 50 --schedule 30  --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/  --opt_method sgd  --train-batch 128 --lr 0.05 --momentum 0.7 --manualSeed 7000 --gpu_id 0


python train_momentsp99.py -a alexnet  --dataset cifar10 --drop 0.3 --epochs 200 --schedule 100 150 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/alexnet200 --opt_method sgd  --train-batch 128 --lr 0.05 --momentum 0.9 --manualSeed 5000 --gpu_id 0

#python train_moment.py -a alexnet  --dataset cifar10 --drop 0.3 --epochs 100 --schedule 50 75 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/alexnet100 --opt_method sgd  --train-batch 128 --lr 0.01 --momentum 0.9 --manualSeed 1000 --gpu_id 0
#python train_moment.py -a alexnet  --dataset cifar10 --drop 0.3 --epochs 100 --schedule 50 75 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/alexnet100 --opt_method sgd  --train-batch 128 --lr 0.01 --momentum 0.9 --manualSeed 3000 --gpu_id 0
#python train_moment.py -a alexnet  --dataset cifar10 --drop 0.3 --epochs 100 --schedule 50 75 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/alexnet100 --opt_method sgd  --train-batch 128 --lr 0.01 --momentum 0.9 --manualSeed 5000 --gpu_id 0 
#cifar100
#python train_pid.py -a densenet --depth 100 --growthRate 12 --dataset cifar100 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-100-12/pid --opt_method sgd --train-batch 128 --I 10 --D 0 --gpu_id 0

