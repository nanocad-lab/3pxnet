# 3PXNet Compiler

Given a trained neural network stored in NNEF format, compile it to C headers and source code using 3PXNet inference library.

## Train a neural network

Please see the training folder in this repo. After training, a folder containing the result of training stored in NNEF format should be in this repo.
All prerequisites to run the compiler are also specified in the Makefile of the training folder.

## Using the compiler

```
python compiler.py --input=<a folder containing a neural network stored in nnef format, its name should end with .nnef> 
--dataset=<choose a dataset to test compiled C source code, please choose from MNIST or CIFAR-10>
--test_start_id=<starting index you want from the dataset>
--test_end_id=<ending index you want from the dataset>
```

## Output

All output files are contained in a folder named "autogen" in this repo.
An executable file called "source" is also contained in this repo.
