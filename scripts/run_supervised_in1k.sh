#!/bin/bash
data="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output"
exp="supervised_in1k_converted/voc07"
config="configs/benchmark_tasks/image_classification/voc07/resnet50_supervised_extract_features.yaml"
params="https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_in1k_supervised_converted.pkl"
blob="res4_5_branch2c_bn_s3k8_resize"

# extract training features
if false; then
mkdir -p $data/extract_features/$exp
python tools/extract_features.py \
    --config_file $config \
    --data_type train \
    --output_file_prefix trainval \
    --output_dir $data/extract_features/$exp \
    TEST.PARAMS_FILE $params \
    TRAIN.DATA_FILE $data/voc07/train_images.npy \
    TRAIN.LABELS_FILE $data/voc07/train_labels.npy

# extract testing features
python tools/extract_features.py \
    --config_file $config \
    --data_type test \
    --output_file_prefix test \
    --output_dir $data/extract_features/$exp \
    TEST.PARAMS_FILE $params \
    TEST.DATA_FILE $data/voc07/test_images.npy \
    TEST.LABELS_FILE $data/voc07/test_labels.npy
fi

# train svm
mkdir -p $data/svm/$exp
python tools/svm/train_svm_kfold_parallel.py \
    --data_file $data/extract_features/$exp/trainval_${blob}_features.npy \
    --targets_data_file $data/extract_features/$exp/trainval_${blob}_targets.npy \
    --costs_list "1.0,10.0,100.0" \
    --output_path $data/svm/$exp

python tools/svm/test_svm.py \
  --data_file $data/extract_features/$exp/test_${blob}_features.npy \
  --targets_data_file $data/extract_features/$exp/test_${blob}_targets.npy \
  --costs_list "1.0,10.0,100.0" \
  --output_path $data/svm/$exp
