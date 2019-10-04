#!/bin/bash

model=$1
python extra_scripts/pickle_caffe2_to_pytorch.py \
    --c2_model $model \
    --output_model ${model}.pth \
    --arch "R-50"
    #--bgr2rgb \
    #--repeat_conv1
