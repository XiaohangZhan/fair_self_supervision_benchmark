#!/bin/bash

# firstly, download VOC2007 trainval and test and decompress them.

data="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output"
VOC="/mnt/lustre/share/zhanxiaohang/data/VOCdevkit/VOC2007/"

mkdir -p $data/voc07/low_shot/labels/
python extra_scripts/create_voc_data_files.py \
    --data_source_dir $VOC \
    --output_dir $data/voc07/ \
    --generate_json 1

python extra_scripts/create_voc_low_shot_challenge_samples.py \
    --targets_data_file $data/voc07/train_targets.json \
    --output_path $data/voc07/low_shot/labels/ \
    --k_values "1,2,4,8,16,32,64,96" \
    --num_samples 5

mkdir $VOC/lists

awk 'NF{print $0 ".jpg"}' $VOC/ImageSets/Main/trainval.txt $VOC/ImageSets/Main/test.txt > $VOC/lists/trainvaltest.txt
