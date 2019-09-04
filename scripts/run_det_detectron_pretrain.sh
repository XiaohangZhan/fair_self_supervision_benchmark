#!/bin/bash
data="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output"
param_folder="/mnt/lustre/share/zhanxiaohang/data/fair_self_supervision_benchmark_cache/models"
detectron="/mnt/lustre/share/zhanxiaohang/proj/detectron"
exp="supervised_in1k/voc07"

python $detectron/tools/train_net.py \
    --multi-gpu-testing \
    --cfg "configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval.yaml" \
    OUTPUT_DIR $data/detectron-output/$exp \
    TRAIN.WEIGHTS /mnt/lustre/share/zhanxiaohang/data/detectron-download-cache/ImageNetPretrained/MSRA/resnet50_in1k_supervised.pkl \
    2>&1 | tee $data/detectron-output/$exp/log.txt
