#!/bin/bash
data="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output"
exp="supervised_in1k/voc07low"
config="configs/benchmark_tasks/low_shot_image_classification/voc07/resnet50_supervised_low_shot_extract_features.yaml"
params="https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_in1k_supervised.pkl"
blob="res4_5_branch2c_bn_s3k8_resize"

if false; then
# train svm
mkdir -p $data/svm/$exp
for s in {1..5}; do
    for k in 1 2 4 8 16 32 64 96; do
        python tools/svm/train_svm_low_shot.py \
            --data_file $data/extract_features/supervised_in1k/voc07/trainval_${blob}_features.npy \
            --targets_data_file $data/voc07/low_shot/labels/train_labels_sample${s}_k${k}.npy \
            --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
            --output_path $data/svm/$exp
    done
done
fi

if true; then
python tools/svm/test_svm_low_shot.py \
  --data_file $data/extract_features/supervised_in1k/voc07/test_${blob}_features.npy \
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
