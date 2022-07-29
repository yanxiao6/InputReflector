#!/bin/bash

 data="cifar10 mnist fmnist"
 model="conv vgg16 resnet"
 gpu=("0" "1" "2" "3")
 tr=("zoom" "blur" "bright" "contrast" "shear" "translation")
 out="svhn fmnist mnist"
 var="true false"


 stage="sia"
 for k in $var:
 do
     for j in $data;
     do
         for i in $model;
         do
             python eval.py -data $j -model $i -trans ${tr[0]} -out $out -gpu ${gpu[0]} -is_diff_data $k -stage $stage &>> logs/eval_sia_$j$i${tr[0]}.txt &
             python eval.py -data $j -model $i -trans ${tr[1]} -out $out -gpu ${gpu[1]} -is_diff_data $k -stage $stage &>> logs/eval_sia_$j$i${tr[1]}.txt &
             python eval.py -data $j -model $i -trans ${tr[2]} -out $out -gpu ${gpu[2]} -is_diff_data $k -stage $stage &>> logs/eval_sia_$j$i${tr[2]}.txt &
             python eval.py -data $j -model $i -trans ${tr[3]} -out $out -gpu ${gpu[3]} -is_diff_data $k -stage $stage &>> logs/eval_sia_$j$i${tr[3]}.txt &
             wait
         done
         wait
     done
     wait
 done


 stage="quad"
 for k in $var:
 do
     for j in $data;
     do
         for i in $model;
         do
             python eval.py -data $j -model $i -trans ${tr[0]} -out $out -gpu ${gpu[0]} -is_diff_data $k -stage $stage &>> logs/eval_quad_$j$i${tr[0]}.txt &
             python eval.py -data $j -model $i -trans ${tr[1]} -out $out -gpu ${gpu[1]} -is_diff_data $k -stage $stage &>> logs/eval_quad_$j$i${tr[1]}.txt &
             python eval.py -data $j -model $i -trans ${tr[2]} -out $out -gpu ${gpu[2]} -is_diff_data $k -stage $stage &>> logs/eval_quad_$j$i${tr[2]}.txt &
             python eval.py -data $j -model $i -trans ${tr[3]} -out $out -gpu ${gpu[3]} -is_diff_data $k -stage $stage &>> logs/eval_quad_$j$i${tr[3]}.txt &
             wait
         done
         wait
     done
     wait
 done


 stage="quad"
 aug=False
 for k in $var:
 do
     for j in $data;
     do
         for i in $model;
         do
             python eval.py -data $j -model $i -trans ${tr[0]} -out $out -gpu ${gpu[0]} -is_diff_data $k -data_aug $aug -stage $stage &>> logs/eval_quad_noaug_$j$i${tr[0]}.txt &
             python eval.py -data $j -model $i -trans ${tr[1]} -out $out -gpu ${gpu[1]} -is_diff_data $k -data_aug $aug -stage $stage &>> logs/eval_quad_noaug_$j$i${tr[1]}.txt &
             python eval.py -data $j -model $i -trans ${tr[2]} -out $out -gpu ${gpu[2]} -is_diff_data $k -data_aug $aug -stage $stage &>> logs/eval_quad_noaug_$j$i${tr[2]}.txt &
             python eval.py -data $j -model $i -trans ${tr[3]} -out $out -gpu ${gpu[3]} -is_diff_data $k -data_aug $aug -stage $stage &>> logs/eval_quad_noaug_$j$i${tr[3]}.txt &
             wait
         done
         wait
     done
     wait
 done

 python collect_auroc_sia.py &>> "logs/logs_collect.txt"
 python search_threshold_quad.py &>> "logs/logs_search.txt"
python search_threshold_quad.py &>> "logs/logs_search_noaug.txt"



# stage="sia"
# adv=True
# for k in $var:
# do
#     for j in $data;
#     do
#         for i in $model;
#         do
#             python eval.py -data $j -model $i -trans ${tr[0]} -out $out -gpu ${gpu[0]} -is_diff_data $k -stage $stage -data_aug_adv $adv &>> logs/eval_adv_sia_$j$i${tr[0]}.txt &
#             python eval.py -data $j -model $i -trans ${tr[1]} -out $out -gpu ${gpu[1]} -is_diff_data $k -stage $stage -data_aug_adv $adv &>> logs/eval_adv_sia_$j$i${tr[1]}.txt &
#             python eval.py -data $j -model $i -trans ${tr[2]} -out $out -gpu ${gpu[2]} -is_diff_data $k -stage $stage -data_aug_adv $adv &>> logs/eval_adv_sia_$j$i${tr[2]}.txt &
#             python eval.py -data $j -model $i -trans ${tr[3]} -out $out -gpu ${gpu[3]} -is_diff_data $k -stage $stage -data_aug_adv $adv &>> logs/eval_adv_sia_$j$i${tr[3]}.txt &
#             wait
#         done
#         wait
#     done
#     wait
# done


# stage="quad"
# adv=True
# for k in $var:
# do
#     for j in $data;
#     do
#         for i in $model;
#         do
#             python eval.py -data $j -model $i -trans ${tr[0]} -out $out -gpu ${gpu[0]} -is_diff_data $k -stage $stage -data_aug_adv $adv &>> logs/eval_adv_quad_$j$i${tr[0]}.txt &
#             python eval.py -data $j -model $i -trans ${tr[1]} -out $out -gpu ${gpu[1]} -is_diff_data $k -stage $stage -data_aug_adv $adv &>> logs/eval_adv_quad_$j$i${tr[1]}.txt &
#             python eval.py -data $j -model $i -trans ${tr[2]} -out $out -gpu ${gpu[2]} -is_diff_data $k -stage $stage -data_aug_adv $adv &>> logs/eval_adv_quad_$j$i${tr[2]}.txt &
#             python eval.py -data $j -model $i -trans ${tr[3]} -out $out -gpu ${gpu[3]} -is_diff_data $k -stage $stage -data_aug_adv $adv &>> logs/eval_adv_quad_$j$i${tr[3]}.txt &
#             wait
#         done
#         wait
#     done
#     wait
# done

# # python collect_auroc_sia.py | tee "logs/logs_collect_adv.txt"
# # python search_threshold_quad.py | tee "logs/logs_search_adv.txt"