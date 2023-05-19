# SADA

This is the pytorch version code of Few-Shot Learning with Visual Distribution Calibration and Cross-Modal Distribution Alignment in CVPR2023.

## Content

- [SADA](#sada)
  - [Content](#content)
  - [Introduce of SADA](#introduce-of-sada)
  - [Datasets](#datasets)
  - [Pretrained CLIP](#pretrained-clip)
  - [Insturction](#insturction)

## [Introduce of SADA](#Content)

SADA is introduced from《Few-Shot Learning with Visual Distribution Calibration and Cross-Modal Distribution Alignment》

Paper ：Runqi Wang, Yuxiang Bao, Baochang Zhang, Jianzhuang Liu, Wentao Zhu, Guodong Guo."Few-Shot Learning with Visual Distribution Calibration and Cross-Modal Distribution Alignment". In CVPR, 2023.

## [Datasets](#Content)

The test Datasets：CIFAR10, [Download link](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)  
Datasets size：10 classes and 32*32 pixels for each image.  
Training set：50,000 images.  
Testing set：10,000 images.


## [Pretrained CLIP](#Content)

We use the pretrained CLIP model from [here](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt)

## [Insturction](#Content)

```python
pip install -r requirements.txt
sh run.sh
```