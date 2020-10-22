# Training

This repository enables the reproduction of the accuracy results reported in the article:
[3PXNet: Pruned-Permuted-Packed XNOR Networks for Edge Machine Learning](url)
The code is based on https://github.com/itayhubara/BinaryNet.pytorch

## Requirements

* Python 3.6, Numpy, pandas, bokeh
* PyTorch 0.4.0 or newer
* Download convert.py from https://github.com/KhronosGroup/NNEF-Tools/tree/master/nnef_tools and put it in this directory

## MNIST

```bash
python main_mnist.py --size 1 --permute 1
```
Trains a small pruned binarized MLP on MNIST with permutation

## CIFAR10 & SVHN

```bash
python main_conv.py --size 1 --permute 1 --dataset CIFAR10
```
Trains a large pruned binarized CNN on CIFAR10 with permutation
