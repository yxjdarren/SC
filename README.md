# Socialized Coevolution: Advancing a Better World through Cross-Task Collaboration
The code repository for "Socialized Coevolution: Advancing a Better World through Cross-Task Collaboration" (the paper has been accepted by ICML 2025) in PyTorch.

## Prerequisites

The following packages are required to run the scripts:

Please see [requirements.txt](./requirements.txt).

## Dataset
We provide the benchmark dataset, i.e., CIFAR100[1] and VOC07+12[2-3]. 

[1] Krizhevsky A, Hinton G. Learning multiple layers of features from tiny images.(2009)[EB/OL].(2009-9)

[2] Everingham M, Van Gool L, Williams C K I, et al. The pascal visual object classes (voc) challenge[J]. International journal of computer vision, 2010, 88(2): 303-338.

[3] Everingham M, Eslami S M A, Van Gool L, et al. The pascal visual object classes challenge: A retrospective[J]. International journal of computer vision, 2015, 111(1): 98-136.

## Testing scripts

- Test CIFAR100 (DISC)
  ```
  python main-cls.py
  ```
  
  Remember to change `--checkpoint` into your own root, or you will encounter errors.

- Test VOC07+12 (DISC)
  ```
  python main-det.py
  ```
  
  Remember to change `--checkpoint` into your own root, or you will encounter errors.

## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- https://huggingface.co/ynhe/INTERN1.0/tree/main

## Contact 
If there are any questions, please feel free to contact with the author:  Xinjie Yao (yaoxinjie@tju.edu.cn). Enjoy the code.
