ifndef PYTHON_EXECUTABLE
    PYTHON_EXECUTABLE = python
endif
prereq:
	pip3 install git+https://github.com/KhronosGroup/NNEF-Tools.git
	pip3 install future typing six numpy protobuf onnx torch
	pip3 install pandas
	pip3 install bokeh
train:
ifdef MNIST
	$(PYTHON_EXECUTABLE) main_mnist.py --size 1 --permute 1
else ifdef CIFAR
	$(PYTHON_EXECUTABLE) main_conv.py --size 1 --permute 1
endif
convert:
ifdef MNIST
	$(PYTHON_EXECUTABLE) convert.py --input-format onnx --output-format nnef --input-mode "training_data/FC_Large.onnx" --output-mode "../FC_Large.nnef"
	$(PYTHON_EXECUTABLE) ../3pxnet-compiler/compiler.py --input="../FC_Large.nnef"
else ifdef CIFAR
	$(PYTHON_EXECUTABLE) convert.py --input-format onnx --output-format nnef --input-mode "training_data/CNN_Large.onnx" --output-mode "../CNN_Large.nnef"
	$(PYTHON_EXECUTABLE) ../3pxnet-compiler/compiler.py --input="../CNN_Large.nnef" --dataset=CIFAR10
endif
clean:
	rm -rf *.nnef *.onnx ../3pxnet-compiler/autogen/*
