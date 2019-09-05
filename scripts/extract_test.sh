#!/bin/bash
srun -p AD -n1 --gres=gpu:8 --ntasks-per-node 1 \
python tools/extract_features.py \
    --config_file $1 \
    --data_type test \
    --output_file_prefix test \
    --output_dir /mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output/extract_features/$2 \
    TEST.DATA_FILE /mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output/voc07/test_images.npy \
    TEST.LABELS_FILE /mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output/voc07/test_labels.npy
