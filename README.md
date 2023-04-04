# DrLim with some comparison

The base model is from [Dimensionality Reduction by Learning an Invariant Mapping CVPR2006](https://ieeexplore.ieee.org/abstract/document/1640964). Implementation is from [heekhero](https://github.com/heekhero/DrLIM). This repo uses Fashion-Mnist, Cifra-10 and Cifar-100 to test the performance of DrLim. As the paper [FaceNet CVPR2015](https://arxiv.org/abs/1503.03832) proposed the triplet loss, contrastive loss is also compared with triplet loss, while they share the same network structure to learn a metric mapping function.

## contrastive loss on Fashion-Mnist
|Fashion-Mnist 2d|Fashion-Mnist 3d|
|:---:|:---:|
|![Fashion-Mnist 2d](./images/2023-04-04_12%3A00%3A39_FashionMnist_i1o2.png)|![Fashion-Mnist 3d](./images/2023-04-04_12%3A08%3A25_FashionMnist_i1o3.png)|


## contrastive loss on Cifar10(RGB)
|Cifar10 2d|Cifar10 3d|
|:---:|:---:|
|![Cifar10 2d](./images/2023-04-04_17%3A33%3A22_Cifar10_i1o2.png)|![Cifar10 3d]()|
haven't found the reason of that outliers which locates far from origin...

## contrastive loss on Cifar10(Gray-scale)
todo...
