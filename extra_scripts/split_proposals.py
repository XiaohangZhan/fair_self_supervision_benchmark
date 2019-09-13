import pickle
import numpy as np

fn = '/DATA/xhzhan/VOC_official/VOCdevkit/proposals/selective_search_msra_voc_2007_trainval.pkl'
train_fn = '/DATA/xhzhan/VOC_official/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
val_fn = '/DATA/xhzhan/VOC_official/VOCdevkit/VOC2007/ImageSets/Main/val.txt'

with open(fn, 'rb') as f:
    prop = pickle.load(f, encoding='latin1')

with open(train_fn, 'r') as f:
    lines = f.readlines()
train_idx = [int(l.strip()) for l in lines]
with open(val_fn, 'r') as f:
    lines = f.readlines()
val_idx = [int(l.strip()) for l in lines]

train_prop = {'boxes': [], 'indexes': [], 'scores': []}
val_prop = {'boxes': [], 'indexes': [], 'scores': []}

belong_dict = {}
for ti in train_idx:
    belong_dict[ti] = 0
for vi in val_idx:
    belong_dict[vi] = 1

for i in range(len(prop['indexes'])):
    ind = prop['indexes'][i]
    if belong_dict[ind] == 0:
        train_prop['boxes'].append(prop['boxes'][i])
        train_prop['indexes'].append(prop['indexes'][i])
        train_prop['scores'].append(prop['scores'][i])
    else:
        val_prop['boxes'].append(prop['boxes'][i])
        val_prop['indexes'].append(prop['indexes'][i])
        val_prop['scores'].append(prop['scores'][i])

assert (np.array(train_prop['indexes']) == np.array(train_idx)).all()
assert (np.array(val_prop['indexes']) == np.array(val_idx)).all()

with open(fn.replace('trainval', 'train'), 'wb') as f:
    pickle.dump(train_prop, f)
with open(fn.replace('trainval', 'val'), 'wb') as f:
    pickle.dump(val_prop, f)
