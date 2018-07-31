# mxnet-dsod
Under development.This is re-implementations of [DSOD and grp-DSOD](https://github.com/szq0214/DSOD),which train object detector from scratch.
More information from [DSOD: Learning Deeply Supervised Object Detectors from Scratch](https://arxiv.org/pdf/1708.01241.pdf) and [Learning Object Detectors from Scratch with Gated Recurrent Feature Pyramids](https://arxiv.org/pdf/1712.00886.pdf)

## Prerequisites
1. Python 3.6
2. [Mxnet](https://mxnet.apache.org/)
3. Numpy
4. Opencv-python

## Preparations
1. Clone this repository.
2. Download VOC dataset from this [released page](http://host.robots.ox.ac.uk/pascal/VOC).Make .rec file using im2rec.py and put them in floder data if you want to train on VOC.[A tutorial](https://github.com/leocvml/mxnet-im2rec_tutorial)

## Train the model

```
python train.py
# see advanced arguments for training
python train.py -h
```

## TODO
1. Mutil-GPU support.I just have one gpu now,so I don't konw whether the [Synchronized Batch Normalization](https://github.com/zhanghang1989/MXNet-Gluon-SyncBN) works well or not.I will updata it when I can use more gpu.Or you can modify it by youself.
2. A pretrained model.DSOD and Grp-dsod converge very slowly,the author gets a good performance by training them 100000 epochs on VOC.It will take a long time.And I will train it if my gpu is free.Don't expect too much.
3. Training on a small dataset.
