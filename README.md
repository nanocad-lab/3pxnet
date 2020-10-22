# 3PXNet
This is a software library used for training and inference of a neural network of 3PXNet style.
## Directory Structure
```
.
├── 3pxnet-inference        # Inference library
    ├── 3pxnet              # 3PXNet library
        ├── 3pxnet_cn.c/h   # 3PXNet convolutional primitives
        ├── 3pxnet_fc.c/h   # 3PXNet fully-connected primitives
    ├── bwn                 # Binary-weight functions for CNNS
        ├── bwn_dense_cn.c/h# BWN Convolutional Layer implementation
    ├── examples            # Example inference programs
    ├── log                 # Validation logfiles
    ├── scripts             # Misc. scripts (python)
    ├── util                # Support libraries for 3PXNet and XNOR primitives
        ├── datatypes.h     # Datatypes used by XNOR and 3PXNet primitives
        ├── utils.h         # Macros used by XNOR and 3PXNet primitives
        ├── xnor_base.h     # Functions used by XNOR and 3PXNet primitives
    ├── val                 # Validation functions
    ├── xnor                # XNOR library
        ├── xnor_cn.c/h     # 3PXNet convolutional primitives
        ├── xnor_fc.c/h     # 3PXNet fully-connected primitives
    ├── doxygen.cfg         # Doxygen configuration
    ├── LICENSE.md          # License information
    ├── Makefile            # Project Makefile
    └── README.md           # Detailed introduction to inference library
├── 3pxnet-training         # Training library
    ├──Makefile             # Project Makefile
    ├──README.md            # Detailed introduction to training library
    ├──main_conv.py         # Train a CNN in 3PXNet style
    ├──main_mnist.py        # Train a MLP in 3PXNet style
    ├──binarized_modules.py # Definition for single binarized layer
    ├──network.py           # Definition for networks able to be trained
    ├──utils.py             # Definition for miscellaneous functions
    ├──utils_own.py         # Same as above
├── 3pxnet-compiler         # Compiler for the whole project
    ├── compiler.py        # The compiler
    ├── test.py             # Testing script for compiler
    ├── utils.py            # Needed for an import function in converter.py
    ├── utils_own.py        # Same as above
```
## Prerequisites
### Training and Compiling
python packages: future typing six numpy protobuf onnx torch nnef pandas bokeh

To install them all, go to 3pxnet-training directory and type
```
make prereq
```
After that, to install package nnef, please go to https://github.com/KhronosGroup/NNEF-Tools/tree/master/parser and follow the instructions there.

If you want to use your own python executable, please define it as PYTHON_EXECUTABLE. Otherwise, makefile later will use 'python' as default.
### Inference

Building the library and examples on an X86 machine requires the following:
* GCC Compiler (tested on version 6.5.0)
* GNU Make (tested on version 4.1)
* Python 2/3 for running validation test suite (tested with versions 2.7.12 and 3.5.2)
* (optional) gzip for generating new MNIST headers for the examples (tested with version 1.6)
* (optional) Doxygen for generating documentation (tested with version 1.8.11)

## Usage
To train a neural network in 3PXNet style, go to 3pxnet-training directory and type either
```
make MNIST=1 train
```
or
```
make CIFAR=1 train
```
Next, we need to convert the trained network into NNEF format, so please go to https://github.com/KhronosGroup/NNEF-Tools/tree/master/nnef_tools, download convert.py, and put it in 3pxnet-training directory.

To use this file and compile the NNEF formatted network, go to 3pxnet-training and type
```
make MNIST=1 convert
```
or
```
make CIFAR=1 convert
```
The compiled result is in a directory called "autogen" in 3pxnet-compiler.

To test the compiled result, go to 3pxnet-inference and type
```
make test
```
If you would like to train a permuted network on your own, then please export the permutation list for each layer and name them as fc_#_list.npy for a fully connected layer,
and conv_#_list.npy for a convolution layer. After this exportation, put them in the NNEF directory which should contain your trained network in NNEF format.
Besides that, in case you want to test the compiled code, you should create a file called __Golden.txt, which contains the golden output obtained from training result.
This first three lines of this text file should can be anything, the fourth line should have all those golden labels on a single line with one space separating them.

If you want to compile your own version of neural network stored in NNEF format, you can do so by going to 3pxnet-compiler directory and use this command:
```
python compiler.py --input=<your directory containing all NNEF formatted neural network, should be named like xxxx.nnef> --dataset=<the dataset you used for training>
```
Please note that not every neural network in NNEF can be compiled. A network must be binarized and have only padding, pooling, and batch normalization operations besides convolution and matrix multiplication to be compiled. The number of kernels in a convolutional layer must be a multiple of 32 otherwise it is not supported by the inference library.

Also note that the dataset here is used for testing and example inputs. Currently, only MNIST and CIFAR10 is supported, but if you want to use a different one, you can check
a function in compiler.py called write_images, it provides an example telling you for MNIST and CIFAR10, how are the images and labels loaded, and you can modify that to
accomodate for your dataset.

To run the executable file, go to 3pxnet-copiler directory and type
```
./source
```