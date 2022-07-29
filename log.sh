#!/bin/bash

# train original models
tr="blur"
data="cifar10 mnist fmnist"
model=("conv" "vgg16")
gpu=("0" "1" "2" "3")

aug=False
adv=False
for i in $data;
do
    python train_model.py -gpu ${gpu[0]} -dataset $i -model ${model[0]} -trans $tr | tee logs/train_$i${model[0]}.txt &
    python train_model.py -gpu ${gpu[2]} -dataset $i -model ${model[1]} -trans $tr | tee logs/train_$i${model[1]}.txt &
    python cifar_resnet.py $i $tr $aug $adv ${gpu[3]} | tee logs/train_resnet_$i.txt &
    wait
done

# degree
tr=("zoom" "blur" "bright" "contrast" "shear" "translation")
data="cifar10 mnist fmnist"
model="conv vgg16 resnet"
gpu=("0" "1" "2" "3")

for j in $data;
do
    for i in $model;
    do
        python seek_degree.py -a ${tr[0]} -d $j -m $i -gpu ${gpu[0]}  | tee logs/degree_$j$i${tr[0]}.txt &
        python seek_degree.py -a ${tr[1]} -d $j -m $i -gpu ${gpu[1]}  | tee logs/degree_$j$i${tr[1]}.txt &
        python seek_degree.py -a ${tr[2]} -d $j -m $i -gpu ${gpu[2]}  | tee logs/degree_$j$i${tr[2]}.txt &
        python seek_degree.py -a ${tr[3]} -d $j -m $i -gpu ${gpu[3]}  | tee logs/degree_$j$i${tr[3]}.txt &
        wait
    done
    wait
done

# train original models with data augmentation
tr="blur"
model=("conv" "vgg16")
aug=True
adv=False
for i in $data;
do
    python train_model.py -gpu ${gpu[0]} -dataset $i -model ${model[0]} -trans $tr -data_aug $aug | tee logs/train_aug_$i${model[0]}.txt &
    python train_model.py -gpu ${gpu[2]} -dataset $i -model ${model[1]} -trans $tr -data_aug $aug | tee logs/train_aug_$i${model[1]}.txt &
    python cifar_resnet.py $i $tr $aug $adv ${gpu[3]} | tee logs/train_aug_resnet_$i.txt &
    wait
done


# # train original models with data augmentation using adversarial data
# aug=True
# adv=True
# for i in $data;
# do
#     python train_model.py -gpu ${gpu[1]} -dataset $i -model ${model[0]} -trans $tr -data_aug $aug -data_aug_adv $adv | tee logs/train_aug_adv_$i${model[0]}.txt &
#     python train_model.py -gpu ${gpu[2]} -dataset $i -model ${model[1]} -trans $tr -data_aug $aug -data_aug_adv $adv | tee logs/train_aug_adv_$i${model[1]}.txt &
#     python resnet.py $i $tr $aug $adv ${gpu[3]} | tee logs/train_aug_adv_resnet_$i.txt &
#     wait
# done



# train siamese network
tr="blur"
data="cifar10 mnist fmnist"
model=("conv" "vgg16" "resnet")
gpu=("0" "2" "3")

for i in $data;
do
    python train.py -data $i -model ${model[0]} -trans $tr -gpu ${gpu[0]} &> logs/train_sia_$tr$i${model[0]}.txt &
    python train.py -data $i -model ${model[1]} -trans $tr -gpu ${gpu[1]} &> logs/train_sia_$tr$i${model[1]}.txt &
    python train.py -data $i -model ${model[2]} -trans $tr -gpu ${gpu[2]} &> logs/train_sia_$tr$i${model[2]}.txt &
    wait
done


# train quadruptlet network with data augmentation
stage="quad"
for i in $data;
do
    python train.py -data $i -model ${model[0]} -trans $tr -gpu ${gpu[0]} -stage $stage &> logs/train_quad_$tr$i${model[0]}.txt &
    python train.py -data $i -model ${model[1]} -trans $tr -gpu ${gpu[1]} -stage $stage &> logs/train_quad_$tr$i${model[1]}.txt &
    python train.py -data $i -model ${model[2]} -trans $tr -gpu ${gpu[2]} -stage $stage &> logs/train_quad_$tr$i${model[2]}.txt &
    wait
done

# train quadruptlet network without data augmentation
stage="quad"
aug=False
for i in $data;
do
    python train.py -data $i -model ${model[0]} -trans $tr -gpu ${gpu[0]} -stage $stage -data_aug $aug &> logs/train_quad_noaug_$tr$i${model[0]}.txt &
    python train.py -data $i -model ${model[1]} -trans $tr -gpu ${gpu[1]} -stage $stage -data_aug $aug &> logs/train_quad_noaug_$tr$i${model[1]}.txt &
    python train.py -data $i -model ${model[2]} -trans $tr -gpu ${gpu[2]} -stage $stage -data_aug $aug &> logs/train_quad_noaug_$tr$i${model[2]}.txt &
wait
done


# adv=True
# # train siamese network with adversarial
# for i in $data;
# do
#     python train.py -data $i -model ${model[0]} -trans $tr -gpu ${gpu[0]} -data_aug_adv $adv &> logs/train_adv_sia_$tr$i${model[0]}.txt &
#     python train.py -data $i -model ${model[1]} -trans $tr -gpu ${gpu[1]} -data_aug_adv $adv &> logs/train_adv_sia_$tr$i${model[1]}.txt &
#     python train.py -data $i -model ${model[2]} -trans $tr -gpu ${gpu[2]} -data_aug_adv $adv &> logs/train_adv_sia_$tr$i${model[2]}.txt &
#     wait
# done


# # train quadruptlet network with adversarial
# stage="quad"
# for i in $data;
# do
#     python train.py -data $i -model ${model[0]} -trans $tr -gpu ${gpu[0]} -stage $stage -data_aug_adv $adv &> logs/train_adv_quad_$tr$i${model[0]}.txt &
#     python train.py -data $i -model ${model[1]} -trans $tr -gpu ${gpu[1]} -stage $stage -data_aug_adv $adv &> logs/train_adv_quad_$tr$i${model[1]}.txt &
#     python train.py -data $i -model ${model[2]} -trans $tr -gpu ${gpu[2]} -stage $stage -data_aug_adv $adv &> logs/train_adv_quad_$tr$i${model[2]}.txt &
#     wait
# done
