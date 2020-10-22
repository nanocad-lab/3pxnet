# Compiler
GCC = 1
ARMCC = 0
# Word size
PCKW = 32
ARCH = 64
# Intel hw popcount support
INTL = 1
# Force software popcount
PCNTSW = 1
# Neon support
NEON = 0

# Include Directories
INCLUDE_DIRECTORIES = ./util/ ./val/ ./xnor/ ./3pxnet/ ./bwn/ ./examples/ ../3pxnet-compiler/autogen/

# XNOR Library
XNOR_OBJ = xnor_base.o xnor_fc.o 3pxnet_fc.o xnor_cn.o 3pxnet_cn.o bwn_dense_cn.o

# Validation
VAL_OBJ  = fc_reference.o cn_reference.o


# Compiler
CC = gcc
CFLAGS = $(addprefix -I ,$(INCLUDE_DIRECTORIES)) -O3 -std=c99 -lm -g -ffloat-store
ifndef PYTHON_EXECUTABLE
    PYTHON_EXECUTABLE = python
endif
# Add configuration specific options
# HW popcount support
ifeq (1,$(NEON))
CFLAGS += -DNEON -mfpu=neon  
else ifeq (1,$(INTL))
CFLAGS += -mpopcnt 
endif
# Pack width
ifeq (32,$(PCKW))
CFLAGS += -DPCK32
else ifeq (64,$(PCKW))
CFLAGS += -DPCK64
endif
# Architecture
ifeq (32,$(ARCH))
CFLAGS += -DARCH32
else ifeq (64,$(ARCH))
CFLAGS += -DARCH64
endif
# Compiler 
ifeq (1,$(GCC))
CFLAGS += -DGCC
else ifeq (1,$(ARMCC))
CFLAGS += -DARMCC
endif
# SW Popcount (own implementation)
ifeq (1,$(PCNTSW))
CFLAGS += -DPCNTSW
endif

VPATH = $(INCLUDE_DIRECTORIES) 

%.o: %c %h datatypes.h 
	$(CC) -c -o $@ $^ $(CFLAGS)

validation: validation.c $(XNOR_OBJ) $(VAL_OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

mnist_mlp_s_xnor: mnist_mlp_s_xnor.c $(XNOR_OBJ) 
	$(CC) -o $@ $^ $(CFLAGS)
mnist_mlp_s_3pxnet: mnist_mlp_s_3pxnet.c $(XNOR_OBJ) 
	$(CC) -o $@ $^ $(CFLAGS)
mnist_cnn_s_xnor: mnist_cnn_s_xnor.c $(XNOR_OBJ) 
	$(CC) -o $@ $^ $(CFLAGS)
mnist_cnn_s_3pxnet: mnist_cnn_s_3pxnet.c $(XNOR_OBJ) 
	$(CC) -o $@ $^ $(CFLAGS)
autogen: source.c $(XNOR_OBJ)
	$(CC) -o source $^ $(CFLAGS)
clean:
	rm -rf *.o bin_nn validation mnist_mlp_s_xnor mnist_mlp_s_3pxnet mnist_cnn_s_xnor mnist_cnn_s_3pxnet logfile doxygen.warn html/* log/* ../__Compiled.txt ../__Golden.txt
test: clean autogen
	./source | grep 'label:\s' > '__Compiled.txt'
	mv __Compiled.txt ../3pxnet-compiler
	$(PYTHON_EXECUTABLE) ../3pxnet-compiler/test.py
