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
 * \file      mnist_mlp_s_3pxnet.c
 * \brief     Example validation on MNIST using a 3PXNet small MLP
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
#include "xnor_fc.h"
#include "3pxnet_fc.h"

// Weights
#include "mnist_mlp_s_sparse_fc1_wgth.h"
#include "mnist_mlp_s_sparse_fc2_wgth.h"
#include "mnist_mlp_s_sparse_fc1_bn.h"
#include "mnist_mlp_s_sparse_fc2_bn.h"

// Images
#include "mnist_img_lbl.h"

// Network params
// Layer 1
// Input size
#define F1NI   768
// Non-pruned inputs
#define F1BK   64
// Output size
#define F1NO   256
// Layer 1
// Non-pruned inputs
#define F2BK   32
// Output size
#define F2NO   10
// Layer 1 buffers
// Weights
static pckDtype l1_sparse_wgt[]  = MNIST_MLP_S_SPARSE_FC1_WGHT ;
// Indices
static uint8_t  l1_sparse_ind[]  = MNIST_MLP_S_SPARSE_FC1_IND  ;
// Batch norm (threshold/sign)
static bnDtype  l1_sparse_bnth[] = MNIST_MLP_S_SPARSE_FC1_THR  ;
static pckDtype l1_sparse_sign[] = MNIST_MLP_S_SPARSE_FC1_SGN  ;
// Layer 2 buffers
// Weights
static pckDtype l2_sparse_wgt[]  = MNIST_MLP_S_SPARSE_FC2_WGHT  ;
// Indices
static uint8_t  l2_sparse_ind[]  = MNIST_MLP_S_SPARSE_FC2_IND   ;
// Batch Norm
static bnDtype  l2_sparse_mn[]   = MNIST_MLP_S_SPARSE_FC2_MEN ;
static bnDtype  l2_sparse_vr[]   = MNIST_MLP_S_SPARSE_FC2_VAR ;
static bnDtype  l2_sparse_gm[]   = MNIST_MLP_S_SPARSE_FC2_GMM  ;
static bnDtype  l2_sparse_bt[]   = MNIST_MLP_S_SPARSE_FC2_BET ;
// Activation buffers
// L1 Input
static uint8_t    l1_act[]   =  MNIST_IMAGES ;
// L1 Input binarized
static pckDtype  l1_act_bin[F1NI/pckWdt];
// L1 output/ L2 input binarized
static pckDtype   l2_act_bin[F1NO/pckWdt];
// L2 output
static float      output[F2NO];
// Data labels
static uint8_t   labels[] = MNIST_LABELS;

int main( void) {

   // Validation counter
   int correct = 0;

   // Loop through images
   for (int img = 0; img < 100; img++) {
      // Get pointer to current image
      uint8_t *curr_im = l1_act + img*784*sizeof(uint8_t);
      // Pack inputs
      packBinThrsArr(curr_im, l1_act_bin, F1NI, 33);
      // Layer 1
      Fc3pxnWrap(l1_act_bin, l1_sparse_wgt, l1_sparse_ind, F1BK, F1NO, l2_act_bin, l1_sparse_bnth, l1_sparse_sign);
      // Layer 2
      int res = Fc3pxnNoBinWrap(l2_act_bin, l2_sparse_wgt, l2_sparse_ind, F2BK, F2NO, output, l2_sparse_mn, l2_sparse_vr, l2_sparse_gm, l2_sparse_bt);
      // Find maximum label
      float max = -INFINITY;
      int maxIdx = 0;
      for (int i = 0; i <10; i++) {
         if (output[i] > max) {
            max = output[i];
            maxIdx = i;
         }
      }
      printf("Image %d: label: %d, actual: %d\n",img, maxIdx, labels[img]);
      // Check if inference is correct
      if (maxIdx == labels[img]) {
         correct += 1;
      }

   }
   // Print results
   printf("Accuracy: %f%%\n", 100.0*(float)correct/(100.0));


   return (EXIT_SUCCESS);
}
