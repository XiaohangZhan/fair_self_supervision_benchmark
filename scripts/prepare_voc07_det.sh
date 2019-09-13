#!/bin/bash

# firstly, download VOC2007 trainval and test and decompress them.

VOC="/DATA/xhzhan/VOC_official/VOCdevkit/"
detectron="/home/xhzhan/Work_cdc39/proj/detectron"

mkdir -p $detectron/detectron/datasets/data/VOC2007

ln -s $VOC/VOC2007/JPEGImages $detectron/detectron/datasets/data/VOC2007/JPEGImages
ln -s detectron_files/annotations $detectron/detectron/datasets/data/VOC2007/annotations
ln -s $VOC $detectron/detectron/datasets/data/VOC2007/VOCdevkit2007
