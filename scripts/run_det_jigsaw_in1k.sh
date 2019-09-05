#!/bin/bash
data="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output"
param_folder="/mnt/lustre/share/zhanxiaohang/data/fair_self_supervision_benchmark_cache/models"
detectron="/mnt/lustre/share/zhanxiaohang/proj/detectron"
exp="jigsaw_in1k/voc07"


#if true; then
#python extra_scripts/pickle_jigsaw_model.py \
#    --model_file $param_folder/resnet50_jigsaw_in1k_pretext.pkl \
#    --output_file $param_folder/resnet50_jigsaw_in1k_pretext_converted.pkl
#fi
#
#if true; then
#python extra_scripts/pickle_caffe2_detection.py \
#    --c2_model $param_folder/resnet50_jigsaw_in1k_pretext_converted.pkl \
#    --output_file $param_folder/resnet50_jigsaw_in1k_pretext_detectron.pkl \
#    --absorb_std
#fi

mkdir -p $data/detectron-output/$exp

python $detectron/tools/train_net.py \
    --multi-gpu-testing \
    --cfg "configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval.yaml" \
    OUTPUT_DIR $data/detectron-output/$exp \
    TRAIN.WEIGHTS $param_folder/resnet50_jigsaw_in1k_pretext.pkl \
    2>&1 | tee $data/detectron-output/$exp/log.txt
