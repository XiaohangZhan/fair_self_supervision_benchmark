import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True)
args = parser.parse_args()

with open(os.path.join(args.root, 'pascal_train2007.json'), 'r') as f:
    train_annot = json.load(f)
with open(os.path.join(args.root, 'pascal_val2007.json'), 'r') as f:
    val_annot = json.load(f)

train_annot['images'].extend(val_annot['images'])
train_annot['annotations'].extend(val_annot['annotations'])

with open(os.path.join(args.root, 'pascal_trainval2007.json'), 'w') as f:
    json.dump(train_annot, f)
