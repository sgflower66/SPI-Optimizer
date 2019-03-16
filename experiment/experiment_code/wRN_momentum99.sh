#cifar10
python train_momentsp99.py  -a wrn --depth 16 --widen-factor 8 --dataset cifar10 --drop 0.0 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoint/test/ --opt_method sgd  --train-batch 128 --lr 0.18 --manualSeed 20180817 --momentum 0.9 --gpu_id 1 --KD 0 --KDD 0
#python train_momentsp99.py  -a wrn --depth 16 --widen-factor 8 --dataset cifar10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoint/test/ --opt_method sgd  --train-batch 128 --lr 0.7 --manualSeed 20180817 --momentum 0.7 --gpu_id 1 --KD 0 --KDD 0
#cifar100
#python train_pid.py -a densenet --depth 100 --growthRate 12 --dataset cifar100 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-100-12/pid --opt_method sgd --train-batch 128 --I 10 --D 0 --gpu_id 0

