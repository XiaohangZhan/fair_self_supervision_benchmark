#!/bin/bash
work_path=$(dirname $0)
NGPU=2
part=$1

fair_ssl_tools="/mnt/lustre/share/zhanxiaohang/proj/fair_self_supervision_benchmark"
detectron="/mnt/lustre/share/zhanxiaohang/proj/detectron"

pretrain="pretrains/resnet50-19c8e357.pth"

if false; then
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

unset CUDA_VISIBLE_DEVICES
mkdir $work_path/detectron
srun -p $part -n1 --gres=gpu:$NGPU --ntasks-per-node 1 \
python $detectron/tools/train_net.py \
    --multi-gpu-testing \
    --cfg "$fair_ssl_tools/configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval.yaml" \
    OUTPUT_DIR $work_path/detectron/ \
    TRAIN.WEIGHTS ${pretrain}.detectron.pkl \
    2>&1 | tee $work_path/logs/eval_det.log
