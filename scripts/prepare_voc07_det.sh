#!/bin/bash

# firstly, download VOC2007 trainval and test and decompress them.

if [ ! -d "detectron_files" ]; then
    echo "Download detectron_files.tar.gz from https://drive.google.com/open?id=1wIzRu4i36TSWmjxR9lZgH6au01RDjYEl and uncompress it under fair_self_supervision_benchmark"
fi

VOC="/home/xhzhan/data/VOC/VOCdevkit"
SSL="/home/xhzhan/proj/fair_self_supervision_benchmark"
detectron="/home/xhzhan/proj/Detectron"

mkdir -p $detectron/detectron/datasets/data/VOC2007

ln -s $VOC/VOC2007/JPEGImages $detectron/detectron/datasets/data/VOC2007/JPEGImages
ln -s $SSL/detectron_files/annotations $detectron/detectron/datasets/data/VOC2007/annotations
ln -s $VOC $detectron/detectron/datasets/data/VOC2007/VOCdevkit2007
