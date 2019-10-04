#!/bin/bash
pretrain=$1
exp=$2

export CUDA_VISIBLE_DEVICES=0,1,2,3
outdir="/home/xhzhan/data/ssl-benchmark-output/detectron-output"
fair_ssl_tools="/home/xhzhan/proj/fair_self_supervision_benchmark"
detectron="/home/xhzhan/proj/Detectron"

if true; then
# pth to caffe2
python $fair_ssl_tools/extra_scripts/pickle_pytorch_to_caffe2.py \
    --pth_model $pretrain \
    --output_model ${pretrain}.pkl \
    --arch "R-50" \
    --bgr2rgb 1
# convert to detectron model
python $fair_ssl_tools/extra_scripts/pickle_caffe2_detection.py \
    --c2_model ${pretrain}.pkl \
    --output_file ${pretrain}.detectron.pkl \
    --absorb_std
fi

mkdir -p $outdir/$exp
python $detectron/tools/train_net.py \
    --multi-gpu-testing \
    --cfg "$fair_ssl_tools/configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval.yaml" \
    OUTPUT_DIR $outdir/$exp \
    TRAIN.WEIGHTS ${pretrain}.detectron.pkl \
    NUM_GPUS 4 \
    2>&1 | tee $outdir/$exp/log.txt

#python $detectron/tools/train_net.py \
#    --cfg "$fair_ssl_tools/configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval.yaml" \
#    OUTPUT_DIR $outdir/$exp \
#    TRAIN.WEIGHTS ${pretrain}.detectron.pkl \
#    TEST.SCALE 112 \
#    TEST.MAX_SIZE 224 \
#    2>&1 | tee $outdir/$exp/log.txt
