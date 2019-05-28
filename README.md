#  Structured Knowledge Distillation for Semantic Segmentation

This repository contains the source code of our paper, [Structured Knowledge Distillation for Semantic Segmentation](https://arxiv.org/pdf/1903.04197.pdf) (accepted for publication in [CVPR'19](http://cvpr2019.thecvf.com/)).

## Sample results

Demo video for the student net (ESPNet) on Camvid

After distillation with mIoU 65.1:
![image]( https://github.com/irfanICMLL/structure_knowledge_distillation/blob/master/demo/output_sd_esp.gif)

Before distillation with mIoU 57.8:
![image]( https://github.com/irfanICMLL/structure_knowledge_distillation/blob/master/demo/output_base_esp.gif)
  



## Structure of this repository
This repository is organized as:
* [config](/config/) This directory contains the settings.
* [dataset](/dataset/) This directory contains the dataloader for different datasets.
* [network](/network/) This directory contains a model zoo for different seg models.
* [utils](/utils/) This directory contains api for calculating the distillation loss and evaluate the results.

## Performance on the CamVid dataset
We apply the distillation method to training the [ESPnet](https://github.com/sacmehta/ESPNet) and achieves an mIoU of 65.1 on the CamVid test set. We used the dataset splits (train/val/test) provided [here](https://github.com/alexgkendall/SegNet-Tutorial). We trained the models at a resolution of 480x360.

Note: We use 2000 more unlabel data as described in our paper.

| Model | mIOU | 
| --  | -- |
| ESPNet_base | 57.8 |
| ESPNet_ours | 61.4 |
| ESPNet_ours+unlabel data | 65.1 |


## Requirement
python3.5 
pytorch0.41 
ninja 
numpy 
cv2 
Pillow
You can also use this [docker](https://hub.docker.com/r/rainbowsecret/pytorch04/tags/)
We recommend to use [Anaconda](https://conda.io/docs/user-guide/install/linux.html). We have tested our code on Ubuntu 16.04.

## Quick start to eval the model
1. download the [Camvid dataset](https://github.com/alexgkendall/SegNet-Tutorial)
2.python eval_esp.py --method student_esp_d --dataset camvid_light --data_list $PATH_OF_THE_TEST_LIST --data_dir $PATH_OF_THE_TEST_DATA --num_classes 11 --restore-from $PATH_OF_THE_PRETRAIN_MODEL --store-output False

## Model Zoo
Pretrain models can be found in the folder [checkpoint](/checkpoint/)



## Citation
If this code is useful for your research, then please cite our paper.
```
@inproceedings{liu2019structured,
  title={Structured Knowledge Distillation for Semantic Segmentation},
  author={Liu, Yifan and Chen, Ke and Liu, Chris and Qin, Zengchang and Luo, Zhenbo and Wang, Jingdong},
  journal={CVPR},
  year={2019}
}
```


## Train script
Coming soon
