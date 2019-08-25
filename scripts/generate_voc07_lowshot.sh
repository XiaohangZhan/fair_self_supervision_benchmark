#!/bin/bash
data="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output"
mkdir -p $data/voc07/low_shot/labels/
python extra_scripts/create_voc_low_shot_samples.py \
    --targets_data_file $data/voc07/train_labels.npy \
    --output_path $data/voc07/low_shot/labels/ \
    --k_values "1,2,4,8,16,32,64,96" \
    --num_samples 5
