#!/bin/bash
pretrain=$1
exp=$2
MULTI=false

#export CUDA_VISIBLE_DEVICES=0,1,2,3
HOME="/mnt/lustre/share/zhanxiaohang"
outdir="$HOME/data/ssl-benchmark-output/detectron-output"
fair_ssl_tools="$HOME/proj/fair_self_supervision_benchmark"
detectron="$HOME/proj/Detectron"

export PYTHONPATH=$detectron:$PYTHONPATH

if [ ! -f "${pretrain}.detectron.pkl" ]; then
# pth to caffe2
python $fair_ssl_tools/extra_scripts/pickle_pytorch_to_caffe2.py \
    --pth_model $pretrain \
    --output_model ${pretrain}.pkl \
    --arch "R-50"
# convert to detectron model
python $fair_ssl_tools/extra_scripts/pickle_caffe2_detection.py \
    --c2_model ${pretrain}.pkl \
    --output_file ${pretrain}.detectron.pkl
fi

mkdir -p $outdir/$exp
if $MULTI; then
    python $detectron/tools/train_net.py \
        --multi-gpu-testing \
        --cfg "$fair_ssl_tools/configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval_sobel_150k.yaml" \
        OUTPUT_DIR $outdir/$exp \
        TRAIN.WEIGHTS ${pretrain}.detectron.pkl \
        NUM_GPUS 2 \
        2>&1 | tee $outdir/$exp/log.txt
else
    python $detectron/tools/train_net.py \
        --cfg "$fair_ssl_tools/configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval_sobel_150k.yaml" \
        OUTPUT_DIR $outdir/$exp \
        TRAIN.WEIGHTS ${pretrain}.detectron.pkl \
        2>&1 | tee $outdir/$exp/log.txt
fi
