#!/bin/bash
data="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output"
exp="jigsaw_yfcc/voc07"
config="configs/benchmark_tasks/image_classification/voc07/resnet50_jigsaw_extract_features.yaml"

# extract training features
#mkdir -p $data/extract_features/$exp
#python tools/extract_features.py \
#    --config_file $config \
#    --data_type train \
#    --output_file_prefix trainval \
#    --output_dir $data/extract_features/$exp \
#    TEST.PARAMS_FILE https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_jigsaw_yfcc100m_pretext.pkl \
#    TRAIN.DATA_FILE $data/voc07/train_images.npy \
#    TRAIN.LABELS_FILE $data/voc07/train_labels.npy
#
## extract testing features
#python tools/extract_features.py \
#    --config_file $config \
#    --data_type test \
#    --output_file_prefix test \
#    --output_dir $data/extract_features/$exp \
#    TEST.PARAMS_FILE https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_jigsaw_yfcc100m_pretext.pkl \
#    TEST.DATA_FILE $data/voc07/test_images.npy \
#    TEST.LABELS_FILE $data/voc07/test_labels.npy

blob="res4_5_branch2c_bn_s0_s3k8_resize"
# train svm
mkdir -p $data/svm/$exp
python tools/svm/train_svm_kfold.py \
    --data_file $data/extract_features/$exp/trainval_${blob}_features.npy \
    --targets_data_file $data/extract_features/$exp/trainval_${blob}_targets.npy \
    --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
    --output_path $data/svm/$exp

python tools/svm/test_svm.py \
  --data_file $data/extract_features/$exp/test_${blob}_features.npy \
  --targets_data_file $data/extract_features/$exp/test_${blob}_targets.npy \
  --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
  --output_path $data/svm/$exp
