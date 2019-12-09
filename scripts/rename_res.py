from glob import glob
import sys
import shutil
import pdb
exp = sys.argv[1]
fold = "/mnt/lustre/share/zhanxiaohang/data/ssl-benchmark-output/detectron-output/{}/test/voc_2007_test/ResNet50_fast_rcnn".format(exp)
files = glob("{}/*.txt".format(fold))
files = [fn.split('/')[-1] for fn in files]
for fn in files:
    newfn = "{}/comp4_".format(fold) + fn[fn.find('det'):]
    shutil.copyfile("{}/{}".format(fold, fn), newfn)
