#!/bin/bash
exp=$1
MULTI=true

#export CUDA_VISIBLE_DEVICES=0,1,2,3
HOME="/mnt/lustre/share/zhanxiaohang"
outdir="$HOME/data/ssl-benchmark-output/detectron-output"
fair_ssl_tools="$HOME/proj/fair_self_supervision_benchmark"
detectron="$HOME/proj/Detectron"

export PYTHONPATH=$detectron:$PYTHONPATH

mkdir -p $outdir/$exp
if $MULTI; then
    python $detectron/tools/train_net.py \
        --multi-gpu-testing \
        --cfg "$fair_ssl_tools/configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval_150k.yaml" \
        OUTPUT_DIR $outdir/$exp \
        NUM_GPUS 2 \
        SOLVER.BASE_LR: 0.001 \
        2>&1 | tee $outdir/$exp/log.txt
else
    python $detectron/tools/train_net.py \
        --cfg "$fair_ssl_tools/configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval_150k.yaml" \
        OUTPUT_DIR $outdir/$exp \
        SOLVER.BASE_LR: 0.001 \
        2>&1 | tee $outdir/$exp/log.txt
fi
