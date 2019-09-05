#!/bin/bash
data="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output"
exp="jigsaw_yfcc/voc07low"
config="configs/benchmark_tasks/low_shot_image_classification/voc07/resnet50_jigsaw_low_shot_extract_features.yaml"
params="https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_jigsaw_yfcc100m_pretext.pkl"
blob="res4_5_branch2c_bn_s0_s3k8_resize"

if true; then
# extract training features
mkdir -p $data/extract_features/$exp
python tools/extract_features.py \
    --config_file $config \
    --data_type train \
    --output_file_prefix trainval \
    --output_dir $data/extract_features/$exp \
    TEST.PARAMS_FILE $params \
    TRAIN.DATA_FILE $data/voc07/train_images.npy \
    TRAIN.LABELS_FILE $data/voc07/low_shot/labels/train_labels_sample1_k1.npy

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

if true; then
# train svm
mkdir -p $data/svm/$exp
for s in {1..5}; do
    for k in 1 2 4 8 16 32 64 96; do
        python tools/svm/train_svm_low_shot.py \
            --data_file $data/extract_features/$exp/trainval_${blob}_features.npy \
            --targets_data_file $data/voc07/low_shot/labels/train_labels_sample${s}_k${k}.npy \
            --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
            --output_path $data/svm/$exp
    done
done
fi

if true; then
python tools/svm/test_svm_low_shot.py \
  --data_file $data/extract_features/$exp/test_${blob}_features.npy \
  --targets_data_file $data/voc07/test_labels.npy \
  --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
  --output_path $data/svm/$exp \
  --k_values "1,2,4,8,16,32,64,96" \
  --sample_inds "0,1,2,3,4"
fi

python tools/svm/aggregate_low_shot_svm_stats.py \
    --output_path $data/svm/$exp \
    --k_values "1,2,4,8,16,32,64,96" \
    --sample_inds "0,1,2,3,4"
