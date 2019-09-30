#  Structured Knowledge Distillation for Dense Prediction

This repository contains the source code of our paper [Structured Knowledge Distillation for Dense Prediction].
It is an extension of our paper [Structured Knowledge Distillation for Semantic Segmentation](https://arxiv.org/pdf/1903.04197.pdf) (accepted for publication in [CVPR'19](http://cvpr2019.thecvf.com/)).

## Sample results

Demo video for the student net (ESPNet) on Camvid

After distillation with mIoU 65.1:
![image]( https://github.com/irfanICMLL/structure_knowledge_distillation/blob/master/demo/output_sd_esp.gif)

Before distillation with mIoU 57.8:
![image]( https://github.com/irfanICMLL/structure_knowledge_distillation/blob/master/demo/output_base_esp.gif)
 
## Structure of this repository
This repository is organized as:
* [ckpt](/ckpt/) This directory contains the pretrained teacher model and a distilled student model.
* [libs](/libs/) This directory contains the inplaceABNSync modes.
* [dataset](/dataset/) This directory contains the dataloader for different datasets.
* [network](/network/) This directory contains a model zoo for network models.
* [utils](/utils/) This directory contains api for calculating the distillation loss.

## Performance on the Cityscape dataset
We apply the distillation method to training the [PSPNet](https://arxiv.org/abs/1612.01105). We used the dataset splits (train/val/test) provided [here](https://github.com/speedinghzl/pytorch-segmentation-toolbox). We trained the models at a resolution of 512x512.
Pi: Pixel-wise distillation PA: Pair-wise distillation HO: holistic distillation

| Model | Average |
| -- | -- |
| baseline | 69.10 |
| +Pi | 70.51 |
| +Pi+Pa | 71.78 |
| +Pi+Pa+Ho | 74.08 |

## Requirement
python3.5 
pytorch0.4.1 
ninja 
numpy 
cv2 
Pillow
We recommend to use [Anaconda](https://conda.io/docs/user-guide/install/linux.html). We have tested our code on Ubuntu 16.04.

## Quick start to test the model
1. download the [Cityscape dataset](https://www.cityscapes-dataset.com/)
2. sh run_test.sh [you should change the data-dir to your own]. By using our distilled student model, which can be gotten in [ckpt], an mIoU of 73.05 is achieved on the Cityscape test set, and 75.3 on validation set.

| Model | Average | roda | sidewalk | building	wall | fence | pole | trafficlight | trafficsign | vegetation | terrain | sky | person | rider | car | truck | bus | train | motorcycle | bicycle |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| IoU | 73.05 | 97.57 | 78.80 | 91.42 | 50.76 | 50.88 | 60.77 | 67.93 | 73.18 | 92.49 | 70.36 | 94.56 | 82.81 | 61.64 | 94.89 | 60.14 | 66.62 | 59.93 | 61.50 | 71.71 |

## Model Zoo
Pretrain models can be found in the folder [checkpoint](/checkpoint/)

## Train script
If you want to reproduce the ablation study in our paper, please modify is_pi_use/is_pa_use/is_ho_use in the run_train_eval.sh.
sh run_train_eval.sh

## Test script
If you want to test your method on the cityscape test set, please modify the data-dir and resume-from path to your own, then run the test.sh and submit your results to www.cityscapes-dataset.net/submit/ 
sh test.sh

