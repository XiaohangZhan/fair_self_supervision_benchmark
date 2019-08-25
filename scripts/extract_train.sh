#!/bin/bash
srun -p AD -n1 --gres=gpu:8 --ntasks-per-node 1 \
python tools/extract_features.py \
    --config_file $1 \
    --data_type train \
    --output_file_prefix trainval \
    --output_dir /mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output/extract_features/$2 \
    TRAIN.DATA_FILE /mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output/voc07/train_images.npy \
    TRAIN.LABELS_FILE /mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output/voc07/train_labels.npy
