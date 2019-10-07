#!/bin/bash
exp=$1

HOME="/mnt/lustre/share/zhanxiaohang"
outdir="$HOME/data/ssl-benchmark-output/detectron-output"
detectron="$HOME/proj/Detectron"

export PYTHONPATH=$detectron:$PYTHONPATH

python $detectron/tools/reval.py $outdir/$exp/test/voc_2007_test/ResNet50_fast_rcnn --matlab
