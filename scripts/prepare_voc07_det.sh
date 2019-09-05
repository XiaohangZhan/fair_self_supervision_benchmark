#!/bin/bash

# firstly, download VOC2007 trainval and test and decompress them.

data="/mnt/lustre/share/zhanxiaohang/data"
VOC=$data/VOCdevkit
detectron="/mnt/lustre/share/zhanxiaohang/proj/detectron"

if [ ! -d $data/PASCAL_VOC ]; then
    wget https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip
    mv PASCAL_VOC.zip $data
    unzip $data/PASCAL_VOC.zip -d $data/
fi

python extra_scripts/merge_voc_det_trainval.py --root $data/PASCAL_VOC

ln -s $data/PASCAL_VOC/pascal_trainval2007.json $data/PASCAL_VOC/voc_2007_trainval.json
ln -s $data/PASCAL_VOC/pascal_test2007.json $data/PASCAL_VOC/voc_2007_test.json

ln -s $VOC/VOC2007/JPEGImages $detectron/detectron/datasets/data/VOC2007/JPEGImages
ln -s $data/PASCAL_VOC $detectron/detectron/datasets/data/VOC2007/annotations
ln -s $VOC $detectron/detectron/datasets/data/VOC2007/VOCdevkit2007

cp detectron_modify_code/dataset_catalog.py $detectron/detectron/datasets/

