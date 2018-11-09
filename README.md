# Repulsion_Loss

Pytorch implementation of Repulsion Loss as described in [Repulsion Loss: Detecting Pedestrians in a Crowd](https://arxiv.org/abs/1711.07752). The baseline is RetinaNet followed by this [repo](https://github.com/yhenon/pytorch-retinanet). 

## Requirements

- Python3
- Pytorch0.4
- torchvision
- tensorboardX

## Installation

Install packages.

```
sudo apt-get install tk-dev python-tk
pip install cffi
pip install cython
pip install pandas
pip install tensorboardX
```

Build NMS.

```
cd Repulsion_Loss/lib
sh build.sh
```

Create folders.

```
cd Repulsion_Loss
mkdir ckpt mAP_txt summary weight
```

## Datasets
This repo is built for human detection. The popular annotation format for human detection（or pedestrian detection) includes bounding boxes of both human and ignore regions such as [Citypersons](https://arxiv.org/pdf/1702.05693.pdf) and [Crowdhuman](https://arxiv.org/pdf/1805.00123.pdf). You should write them in CSV or TXT files.

### Annotations format
Three examples are as follows:

```
$image_path/img_1.jpg x1 y1 x2 y2 person
$image_path/img_1.jpg x1 y1 x2 y2 ignore
$image_path/img_2.jpg . . . . .
```

Images with more than one bounding box should use one row per box. Labels that we often use are 'person' or 'ignore'. When an image does not contain any bounding box, set them '.'. 

### Label encoding file
A TXT file (classes.txt) is needed to map label to ID. Each line means one label name and its ID. One example is as follows:

```
person 0
```

## Pretrained Model

We use resnet18, 34, 50, 101, 152 as the backbone. You should download them and put them to `/weight`.

- resnet18: [https://download.pytorch.org/models/resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth)
- resnet34: [https://download.pytorch.org/models/resnet34-333f7ec4.pth](https://download.pytorch.org/models/resnet34-333f7ec4.pth)
- resnet50: [https://download.pytorch.org/models/resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)
- resnet101: [https://download.pytorch.org/models/resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
- resnet152: [https://download.pytorch.org/models/resnet152-b121ed2d.pth](https://download.pytorch.org/models/resnet152-b121ed2d.pth)

## Training

```
python train.py --csv_train <$path/train.txt> --csv_val <$path/val.txt> --csv_classes <$path/classes.txt> --depth <50> --pretrained resnet50-19c8e357.pth --model_name <model name to save>
```

## Visualization Result
Baseline

![img1](https://github.com/rainofmine/Repulsion_Loss/blob/master/img/1.png)

Add repulsion loss

![img2](https://github.com/rainofmine/Repulsion_Loss/blob/master/img/2.png)

## Reference

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Repulsion Loss: Detecting Pedestrians in a Crowd](https://arxiv.org/abs/1711.07752)