#cifar10

#adam
python train_adam.py  -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/resnet56_30 --opt_method adam  --train-batch 128 --lr 0.001 --manualSeed 1000 --gpu_id 0
python train_adam.py  -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/resnet56_30 --opt_method adam  --train-batch 128 --lr 0.001 --manualSeed 3000 --gpu_id 0
python train_adam.py  -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/resnet56_30 --opt_method adam  --train-batch 128 --lr 0.001 --manualSeed 5000 --gpu_id 0


#rmsprop

python train_adam.py  -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/resnet56_30 --opt_method rmsprop  --train-batch 128 --lr 0.001 --manualSeed 1000 --gpu_id 0
python train_adam.py  -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/resnet56_30 --opt_method rmsprop  --train-batch 128 --lr 0.001 --manualSeed 3000 --gpu_id 0
python train_adam.py  -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/resnet56_30 --opt_method rmsprop  --train-batch 128 --lr 0.001 --manualSeed 5000 --gpu_id 0


#pid

python train_pid.py -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test  --train-batch 128 --opt_method sgd --lr 0.35 --manualSeed 1000 --I 1 --D 1 --gpu_id 0
python train_pid.py -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test  --train-batch 128 --opt_method sgd --lr 0.35 --manualSeed 3000 --I 1 --D 1 --gpu_id 0
python train_pid.py -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test  --train-batch 128 --opt_method sgd --lr 0.35 --manualSeed 5000 --I 1 --D 1 --gpu_id 0



#addsign

python train_addsign.py -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/ --opt_method Addsign  --train-batch 128 --lr 0.35 --momentum 0.9 --manualSeed 1000 --gpu_id 0
python train_addsign.py -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/ --opt_method Addsign  --train-batch 128 --lr 0.35 --momentum 0.9 --manualSeed 3000 --gpu_id 0
python train_addsign.py -a resnet --depth 56 --dataset cifar10 --drop 0.0 --epochs 60 --schedule 30 45 --wd 1e-4 --gamma 0.1 --checkpoint checkpoint/test/ --opt_method Addsign  --train-batch 128 --lr 0.35 --momentum 0.9 --manualSeed 5000 --gpu_id 0









#cifar100
#python train_pid.py -a densenet --depth 100 --growthRate 12 --dataset cifar100 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-100-12/pid --opt_method sgd --train-batch 128 --I 10 --D 0 --gpu_id 0

