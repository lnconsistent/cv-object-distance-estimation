import os
import torch as t
from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at, inference as inf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# df = pd.DataFrame(columns=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

# img = read_image('/home/olixu/Downloads/test1.png')
# img = img[0:3, 100:img.shape[1]//3+100, 0:img.shape[2]]
# img = t.from_numpy(img)[None]

def gen_depth(img):
    # returns dataframe with image bounding box
    df = pd.DataFrame(columns=['filename', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'])
    img = t.from_numpy(img)[None]

    faster_rcnn = FasterRCNNVGG16()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    trainer.load('/home/olixu/distance-cnn/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth.701052458187')
    opt.caffe_pretrain = False  # this model was trained from torchvision-pretrained model
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
    box_new = np.asarray(_bboxes)
    label_new = at.tonumpy(_labels[0]).reshape(-1)
    score_new = at.tonumpy(_scores[0]).reshape(-1)

    for i in range(box_new.shape[1]):
        df.at[i, 'filename'] = 'file'+str(i)
        df.at[i, 'class'] = label_new[i]
        df.at[i, 'confidence'] = score_new[i]
        # bbox coordinates
        df.at[i, 'xmin'] = box_new[0, i, 1]
        df.at[i, 'ymin'] = box_new[0, i, 0]
        df.at[i, 'xmax'] = box_new[0, i, 3]
        df.at[i, 'ymax'] = box_new[0, i, 2]
    return inf.infer(df)

'''
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

trainer.load('/home/olixu/distance-cnn/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth.701052458187')
opt.caffe_pretrain=False # this model was trained from torchvision-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
box_new = np.asarray(_bboxes)
label_new = at.tonumpy(_labels[0]).reshape(-1)
score_new = at.tonumpy(_scores[0]).reshape(-1)
print(box_new.shape)

for i in range(box_new.shape[1]):
    df.at[i, 'filename'] = i
    df.at[i, 'class'] = label_new[i]
    # bbox coordinates
    df.at[i, 'xmin'] = box_new[0, i, 1]
    df.at[i, 'ymin'] = box_new[0, i, 0]
    df.at[i, 'xmax'] = box_new[0, i, 3]
    df.at[i, 'ymax'] = box_new[0, i, 2]

df.to_csv('test1.csv', index=False)
vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))


plt.savefig("mygraph1.png")
'''