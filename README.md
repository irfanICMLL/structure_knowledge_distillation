#  Structured Knowledge Distillation for Semantic Segmentation

This repository contains the source code of our paper, [Structured Knowledge Distillation for Semantic Segmentation](https://arxiv.org/pdf/1903.04197.pdf) (accepted for publication in [CVPR'19](http://cvpr2019.thecvf.com/)).

## Sample results

Demo video for the student net on Camvid


 ![image]( https://github.com/irfanICMLL/structure_knowledge_distillation/blob/master/demo/output_base_esp.gif)
  ![image]( https://github.com/irfanICMLL/structure_knowledge_distillation/blob/master/demo/output_sd_esp.gif)


## Structure of this repository
This repository is organized as:
* [train](/train/) This directory contains the source code for trainig the ESPNet-C and ESPNet models.
* [test](/test/) This directory contains the source code for evaluating our model on RGB Images.
* [pretrained](/pretrained/) This directory contains the pre-trained models on the CityScape dataset
  * [encoder](/pretrained/encoder/) This directory contains the pretrained **ESPNet-C** models
  * [decoder](/pretrained/decoder/) This directory contains the pretrained **ESPNet** models


## Performance on the CityScape dataset

Our model ESPNet achives an class-wise mIOU of **60.336** and category-wise mIOU of **82.178** on the CityScapes test dataset and runs at 
* 112 fps on the NVIDIA TitanX (30 fps faster than [ENet](https://arxiv.org/abs/1606.02147))
* 9 FPS on TX2
* With the same number of parameters as [ENet](https://arxiv.org/abs/1606.02147), our model is **2%** more accurate

## Performance on the CamVid dataset

Our model achieves an mIOU of 55.64 on the CamVid test set. We used the dataset splits (train/val/test) provided [here](https://github.com/alexgkendall/SegNet-Tutorial). We trained the models at a resolution of 480x360. For comparison  with other models, see [SegNet paper](https://ieeexplore.ieee.org/document/7803544/).

Note: We did not use the 3.5K dataset for training which was used in the SegNet paper.

| Model | mIOU | Class avg. | 
| -- | -- | -- |
| ENet | 51.3 | 68.3 | 
| SegNet | 55.6 | 65.2 | 
| ESPNet | 55.64 | 68.30 | 

## Pre-requisite

To run this code, you need to have following libraries:
* [OpenCV](https://opencv.org/) - We tested our code with version > 3.0.
* [PyTorch](http://pytorch.org/) - We tested with v0.3.0
* Python - We tested our code with Pythonv3. If you are using Python v2, please feel free to make necessary changes to the code. 

We recommend to use [Anaconda](https://conda.io/docs/user-guide/install/linux.html). We have tested our code on Ubuntu 16.04.

## Citation
If ESPNet is useful for your research, then please cite our paper.
```
@inproceedings{mehta2018espnet,
  title={ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation},
  author={Sachin Mehta, Mohammad Rastegari, Anat Caspi, Linda Shapiro, and Hannaneh Hajishirzi},
  booktitle={ECCV},
  year={2018}
}
```


## FAQs

### Assertion error with class labels (t >= 0 && t < n_classes).

If you are getting an assertion error with class labels, then please check the number of class labels defined in the label images. You can do this as:

```
import cv2
import numpy as np
labelImg = cv2.imread(<label_filename.png>, 0)
unique_val_arr = np.unique(labelImg)
print(unique_val_arr)
```
The values inside *unique_val_arr* should be between 0 and total number of classes in the dataset. If this is not the case, then pre-process your label images. For example, if the label iamge contains 255 as a value, then you can ignore these values by mapping it to an undefined or background class as:

```
labelImg[labelImg == 255] = <undefined class id>
```
