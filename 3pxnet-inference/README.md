# 3PXNet Inference 

This is the software library implementing 3PXNet style neural networks, as well as reference, dense, XNOR networks, described in the following paper:

[3PXNet: Pruned-Permuted-Packed XNOR Networks for Edge Machine Learning](url)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Building the library and examples on an X86 machine requires the following:
* GCC Compiler (tested on version 6.5.0)
* GNU Make (tested on version 4.1)
* Python 2/3 for running validation test suite (tested with versions 2.7.12 and 3.5.2)
* (optional) gzip for generating new MNIST headers for the examples (tested with version 1.6)
* (optional) Doxygen for generating documentation (tested with version 1.8.11)

Building for ARM machines:
* Coming soon.

### Building and running validation.

Before building the library make sure you set the appropriate options in the Makefile:
* GCC/ARMCC - Compiler choice
* PCKW - Pack size
* ARCH - Architecture word size
* INTL - Use hardware popcount on X86 processors
* PCNTSW - Force using software popcounts
* NEON - Neon support for ARM architectures (not all fucntions support it yet)

To build the validation suite run:
```
make validation
```

This will build XNOR and 3PXNet libraries and validation routines. To run the validation suite, run:

```
python scripts/runValidation.py
```

Or, if using Python3:
```
python3 scripts/runValidation.py
```

It might take a while to finish. XNOR and 3PXnet layer primitives are tested against reference implementations in val/. All tests should pass. You can check the individual logfiles in the log directory.

## Running the examples 

You can build and run the following examples:

* mnist_mlp_s_xnor - MNIST classification using a dense XNOR small MLP
* mnist_mlp_s_3pxnet - MNIST classification using a 3PXNet small MLP
* mnist_mlp_s_xnor - MNIST classification using a dense XNOR small CNN 
* mnist_mlp_s_3pxnet - MNIST classification using a 3PXNet small CNN

For network description, please refer to the [3PXNet paper](url). To build an example run

```
make <example_name>
```

And to run it:

```
./<example_name>
```

Currently each example runs an inference on 100 images and prints final inference accuracy. To create different headers (more images, different batch), do the following:

* Fetch the MNIST testing dataset:
```
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

* Unpack:
```
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz
```

* Run the following script:
```
python scripts/genMnistHeader.py
```

* The above script can be manually modifed to change the number of images generated in the header, offset (from which image in the dataset to start), output file name, and whether images are normalized or not (CNN examples use normalized inputs, MLP don't).
* After the file is generated, move it to the example folder:
```
mv <generated_header>.h examples/
```
* If the file name changed, adjust it in the source file of the example you want to run.


## Using the library 

### Directory Structure

This project is organized as follows:

```
.
├── 3pxnet                  # 3PXNet library
    ├── 3pxnet_cn.c/h       # 3PXNet convolutional primitives 
    ├── 3pxnet_fc.c/h       # 3PXNet fully-connected primitives 
├── bwn                     # Binary-weight functions for CNNS
    ├── bwn_dense_cn.c/h    # BWN Convolutional Layer implementation
├── examples                # Example inference programs
├── log                     # Validation logfiles 
├── scripts                 # Misc. scripts (python) 
├── util                    # Support libraries for 3PXNet and XNOR primitives
    ├── datatypes.h         # Datatypes used by XNOR and 3PXNet primitives
    ├── utils.h             # Macros used by XNOR and 3PXNet primitives
    ├── xnor_base.h         # Functions used by XNOR and 3PXNet primitives
├── val                     # Validation functions
├── xnor                    # XNOR library
    ├── xnor_cn.c/h         # 3PXNet convolutional primitives 
    ├── xnor_fc.c/h         # 3PXNet fully-connected primitives 
├── doxygen.cfg             # Doxygen configuration 
├── LICENSE.md              # License information
├── Makefile                # Project Makefile 
└── README.md               # You're reading it now!
```

### XNOR/3PXNet libraries

Currently the following primitvies are supported:

* Fully-Connected layers:
   * 3PXNet and XNOR.
   * With/without output binarization.
   * With/without batch norm.
* Convolutional layers:
   * 3PXNet and XNOR.
   * Without output binarization (full-precision outputs are required for the last layer, and that will typically be a FC layer)
   * With/wihtout batch norm.
   * With/without pooling.
   * With/without padding.

There is also limited support for binary-weight convolutional layers, required as a first layer in CNNs. Some of the XNOR/3PXNet primitives have ARM NEON optimized implementations.

If you don't know what exact function you need, you should use one of the following wrappers, which will automatically chose the best implementation based on the parameters:

* CnXnorWrap: XNOR Convolutional layer wrapper
* FcXnorWrap: XNOR Fully-Connected layer wrapper w/ output binarization
* FcXnorNoBinWrap: XNOR Fully-Connected layer wrapper w/o output binarization
* Cn3pxnWrap: 3PXNet Convolutional layer wrapper
* Fc3pxnWrap: 3PXNet Fully-Connected layer wrapper w/ output binarization
* Fc3pxnNoBinWrap: 3PXNet Fully-Connected layer wrapper w/o output binarization

For exact usage of those functions, refer to appropriate source files, documentation (see below) or the examples.

### Generating documentation

To generate doxygen documentation, please run the following:

```
doxygen doxygen.cfg
```

You can view it in the browser of choice by opening html/index.html

## TODO:

Following things are currently being implemented/cleaned/integrated:

* Performance measuring routines.
* Scripts to generate weight/batch norm headers from training results.
* Full ARM NEON support.

## Authors

* **Wojciech Romaszkan** - *Initial work* - [Github](https://github.com/wromaszkan)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments


