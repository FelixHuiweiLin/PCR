# DVC
Code For CVPR2022 paper "[PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning](https://arxiv.org/abs/2304.04408)"

## Usage

### Data preparation
- CIFAR10 & CIFAR100 will be downloaded during the first run. (datasets/cifar10;/datasets/cifar100)
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download, and place it in datasets/mini_imagenet/


### CIFAR-100
```shell
  python general_main.py --data  cifar100 --cl_type nc --agent PCR  --retrieve random --update random --mem_size 1000
 ```

 ### CIFAR-10
```shell
  python general_main.py --data cifar10 --cl_type nc --agent PCR --retrieve random --update random --mem_size 200
 ```
 
 ### Mini-Imagenet
```shell
python general_main.py --data  mini_imagenet --cl_type nc --agent PCR --retrieve random --update random --mem_size 1000
 ```
 
 ## Reference
[online-continual-learning](https://github.com/RaptorMai/online-continual-learning)
