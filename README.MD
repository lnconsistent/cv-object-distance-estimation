# Using CV to detect and estimate distance to landmarks and objects in RGB images

## 1. Introduction


This project combines a regression neural network on bounding boxes for distance estimation, trained on the KITTI dataset, and the Faster R-CNN segmentation and identification neural network, trained on the PASCAL VOC datasets, to attempt to continuously track and estimate the distance of an object from the camera.



## 2. Demo

Download pretrained model for Faster R-CNN from [Google Drive](https://drive.google.com/open?id=1cQ27LIn-Rig4-Uayzy_gH5-cW-NRGVzY) or [Baidu Netdisk( passwd: scxn)](https://pan.baidu.com/s/1o87RuXW). Please use the Pytorch (torchvision) weights.

For a demonstration on images taken from a webcam run
```
python webcam.py --verbose
```
from the base directory. Use verbosity to turn audio reporting of landmarks on or off.
![](https://github.com/lnconsistent/cv-object-distance-estimation/blob/master/results/g1.jpg)
![](https://github.com/lnconsistent/cv-object-distance-estimation/blob/master/results/g2.jpg)

## Acknowledgements
This work builds on and draws inspiration from the following:

- [Yun Chen's simpler pytorch implmentation of Faster R-CNN](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) which is itself a reference to [py-faster-rcnn by Ross Girshick and Sean Bell](https://github.com/rbgirshick/py-faster-rcnn).
- [Harshil Patel's (incomplete) neural network](https://github.com/harshilpatel312/KITTI-distance-estimation), which used bounding boxes and a regression neural network to estimate distance.
