#!/bin/bash
exp="supervised_in1k/voc07"
pretrain="pretrains/resnet50-19c8e357.pth"

unset CUDA_VISIBLE_DEVICES
outdir="/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output/detectron-output"
fair_ssl_tools="/mnt/lustre/share/zhanxiaohang/proj/fair_self_supervision_benchmark"
detectron="/mnt/lustre/share/zhanxiaohang/proj/detectron"

if false; then
# pth to caffe2
python $fair_ssl_tools/extra_scripts/pickle_pytorch_to_caffe2.py \
    --pth_model $pretrain \
    --output_model ${pretrain}.pkl \
    --arch "R-50" \
    --bgr2rgb
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
    2>&1 | tee $outdir/$exp/log.txt
