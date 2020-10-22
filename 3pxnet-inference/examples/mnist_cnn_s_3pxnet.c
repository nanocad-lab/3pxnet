/*
* MIT License
* 
* Copyright (c) 2019 UCLA NanoCAD Laboratory 
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

/*!
 * \file      mnist_cnn_s_3pxnet.c
 * \brief     Example validation on MNIST using a 3PXNet small CNN 
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */


//////////////////////////////
// General Headers
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>


// Datatypes
#include "datatypes.h"
// NN functions
#include "utils.h"
#include "xnor_base.h"
#include "3pxnet_fc.h"
#include "bwn_dense_cn.h"
#include "3pxnet_cn.h"

// Weights
#include "mnist_cnn_s_sparse_cn1_wght.h"
#include "mnist_cnn_s_sparse_cn2_wght.h"
#include "mnist_cnn_s_sparse_cn1_bn.h"
#include "mnist_cnn_s_sparse_cn2_bn.h"
#include "mnist_cnn_s_sparse_fc1_wgth.h"
#include "mnist_cnn_s_sparse_fc1_bn.h"

// Images
// Use binarized inputs
#include "mnist_img_lbl_norm.h"
//#include "src/irina/mnist_img_lbl.h"

// Network params
// Layer 1
// Input XY
#define InXY   28
// Input channels
#define InZ    1 
// Kernel XY
#define C1KXY  5
// Kernel channels
#define C1KZ   32
// Padding
#define C1PD   0
// Pooling
#define C1PL   2
// Layer 2
// Input XY
#define C2XY   12
// Input channels
#define C2Z    32
// Kernel XY
#define C2KXY  5
// Kernel channels
#define C2KZ   32
// Padding
#define C2PD   0
// Pooling
#define C2PL   2
// Non-pruned inputs 
#define C2BK   64
// Layer 3
// Input size
#define F1NI   512
// Non-pruned inputs 
#define F1BK   288
// Output Size
#define Nou    10

// Layer 1 buffers
// Weights
static int8_t   c1_sparse_wgt[]  = MNIST_CNN_S_SPARSE_CN1_WGHT;
// Batch norm (threshold/sign)
static bnDtype  c1_sparse_th[]   = MNIST_CNN_S_SPARSE_CN1_THR;
static pckDtype c1_sparse_sg[]   = MNIST_CNN_S_SPARSE_CN1_SGN;
// Layer 2 bufers
// Weights
static pckDtype c2_sparse_wgt[]  = MNIST_CNN_S_SPARSE_CN2_WGHT;
// Indices
static uint8_t  c2_sparse_ind[]  = MNIST_CNN_S_SPARSE_CN2_IND  ;
// Batch norm (threshold/sign)
static bnDtype  c2_sparse_th[]   = MNIST_CNN_S_SPARSE_CN2_THR;
static pckDtype c2_sparse_sg[]   = MNIST_CNN_S_SPARSE_CN2_SGN;
// Layer 3 bufers
// Weights
static pckDtype f1_sparse_wgt[]  = MNIST_CNN_S_SPARSE_FC1_WGHT  ;
// Indices
static uint8_t  f1_sparse_ind[]  = MNIST_CNN_S_SPARSE_FC1_IND   ;
// Batch norm
static bnDtype  f1_sparse_mn[]   = MNIST_CNN_S_SPARSE_FC1_MEN ;
static bnDtype  f1_sparse_vr[]   = MNIST_CNN_S_SPARSE_FC1_VAR ;
static bnDtype  f1_sparse_gm[]   = MNIST_CNN_S_SPARSE_FC1_GMM  ;
static bnDtype  f1_sparse_bt[]   = MNIST_CNN_S_SPARSE_FC1_BET ;

// Activation buffers
// L1 Input
static uint8_t    l1_act[]   =  MNIST_IMAGES ;
// L2 Input 
static pckDtype   l2_act_bin[C2XY*C2XY*C2Z/pckWdt];
// L3 Input
static pckDtype   l3_act_bin[F1NI/pckWdt];
// Output
static float      output[Nou];
// Labels
static uint8_t   labels[] = MNIST_LABELS;

uint8_t INPUT[] = {
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,1,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,
1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,
1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

int main( void) {

   // Validation counter
   int correct = 0;

   // Loop through images
   for (int img = 0; img < 100; img++) {
      // Get pointer to current image
      uint8_t *curr_im = l1_act + img*784*sizeof(uint8_t);
      // Layer 1 (dense, binary-weight)
      CnBnBwn(curr_im, c1_sparse_wgt, InZ, InXY, InXY, InZ, C1KXY, C1KXY, C1KZ, C1PD, C1PL, l2_act_bin, c1_sparse_th, c1_sparse_sg);
      // Layer 2
      Cn3pxnWrap(l2_act_bin, c2_sparse_wgt, c2_sparse_ind, C2BK, C2Z, C2XY, C2XY, C2Z, C2KXY, C2KXY, C2KZ, l3_act_bin, C2PD, C2PL, c2_sparse_th, c2_sparse_sg);
      // Layer 3
      int res = Fc3pxnNoBinWrap(l3_act_bin, f1_sparse_wgt, f1_sparse_ind, F1BK, Nou, output, f1_sparse_mn, f1_sparse_vr, f1_sparse_gm, f1_sparse_bt);
      // Find maximum label
      float max = -INFINITY;
      int maxIdx = 0;
      for (int i = 0; i <10; i++) {
         if (output[i] > max) {
            max = output[i];
            maxIdx = i;
         }
      }
      // Check if inference is correct
      printf("Image %d: label: %d, actual: %d\n",img, maxIdx, labels[img]);
      if (maxIdx == labels[img]) {
         correct += 1;
      }

   }
   // Print results
   printf("Accuracy: %f%%\n", 100.0*(float)correct/(100.0));


   return (EXIT_SUCCESS);
}
