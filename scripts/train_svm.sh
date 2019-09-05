#!/bin/bash
root="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output"
srun -p AD -n1 \
python tools/svm/train_svm_kfold.py \
    --data_file $root/extract_features/$1/trainval_res4_5_branch2c_bn_s0_s3k8_resize_features.npy \
    --targets_data_file $root/extract_features/$1/trainval_res4_5_branch2c_bn_s0_s3k8_resize_targets.npy \
    --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
    --output_path $root/svm/jigsaw/voc07/
