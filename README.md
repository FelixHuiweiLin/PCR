# PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning
Code For CVPR'2023 paper "[PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning](https://arxiv.org/abs/2304.04408)" and TNNLS'2025 paper "[HPCR: Holistic Proxy-based Contrastive Replay for Online Continual Learning](https://arxiv.org/pdf/2309.15038)"

The framework is based on [online-continual-learning](https://github.com/RaptorMai/online-continual-learning).
- CIFAR10 & CIFAR100 will be downloaded during the first run. (datasets/cifar10;/datasets/cifar100)
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download, and place it in datasets/mini_imagenet/.

PCR is in [https://github.com/FelixHuiweiLin/PCR/blob/main/agents/pcr.py](https://github.com/FelixHuiweiLin/PCR/blob/main/agents/pcr.py).


### CIFAR-100
```shell
  python general_main.py --num_runs 1 --data  cifar100 --cl_type nc --agent PCR  --retrieve random --update random --mem_size 1000 
 ```

 ### CIFAR-10
```shell
  python general_main.py --num_runs 1 --data cifar10 --cl_type nc --agent PCR --retrieve random --update random --mem_size 200
 ```
 
 ### Mini-Imagenet
```shell
python general_main.py --data --num_runs 1  mini_imagenet --cl_type nc --agent PCR --retrieve random --update random --mem_size 1000
 ```

HPCR is in [https://github.com/FelixHuiweiLin/PCR/blob/main/agents/hpcr.py](https://github.com/FelixHuiweiLin/PCR/blob/main/agents/hpcr.py).

### CIFAR-100
```shell
  python general_main.py --num_runs 1 --data  cifar100 --cl_type nc --agent HPCR  --retrieve random --update random --mem_size 1000 --hpcr_temp_max 1.8 --hpcr_temp_min 0.5625 --hpcr_PCD_alpha 0.1 --hpcr_SCD_beta 0.5
 ```

 ### CIFAR-10
```shell
  python general_main.py --num_runs 1 --data cifar10 --cl_type nc --agent HPCR --retrieve random --update random --mem_size 200 --hpcr_temp_max 1.8 --hpcr_temp_min 0.5625 --hpcr_PCD_alpha 0.01 --hpcr_SCD_beta 0.5
 ```
 
 ### Mini-Imagenet
```shell
python general_main.py --data --num_runs 1  mini_imagenet --cl_type nc --agent HPCR --retrieve random --update random --mem_size 1000 --hpcr_temp_max 1.8 --hpcr_temp_min 0.5625 --hpcr_PCD_alpha 0.1 --hpcr_SCD_beta 0.5
 ```

