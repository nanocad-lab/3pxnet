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
 * \file      validation.c
 * \brief     Validates XNOR/3PXNet implementations against reference.
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

//////////////////////////////
// Reference Headers 
#include "fc_reference.h"
#include "cn_reference.h"

//////////////////////////////
// XNOR Headers 
#include "datatypes.h"
#include "xnor_base.h"
// FC
#include "xnor_fc.h"
#include "3pxnet_fc.h"
// CN
#include "xnor_cn.h"
#include "3pxnet_cn.h"

int main(int argc, char** argv) {

   // Time info
   time_t                        rawtime;
   struct                        tm * timeinfo;
   char                          timestr[16];
   // Getopt 
   extern char                   *optarg; 
   extern int                    optind;
   int                           opt;
   // FC/CN layer test
   enum {FC_TEST, CN_TEST}       mode = FC_TEST;
   // Batch Normalization On/Off
   enum {BN_TRUE, BN_FALSE}      bn_mode = BN_FALSE;
   // Output binarization On/Off
   enum {OUT_BIN, NO_OUT_BIN}    out_bin = OUT_BIN;
   // Sparsity On/Off
   enum {SPARSE, DENSE}          sparsity = DENSE;
   int                           pruned = 0;
   // Packing On/Off 
   enum {PACK, UNCONST}          packing = UNCONST;
   // Padding
   enum {NOPAD, PAD}             pad = NOPAD;
   int                           padding = 0;
   // Pooling
   enum {NOPOOL, POOL}           pooling = NOPOOL;
   int                           pool = 1;
   // Logfile
   FILE                          *logFile;
   char                          filename[100] = "";
   // For activation/weight randomization
   int                           xnorVals[2] = {-1,1};
   // BN epsilon
   float                         epsilon = 0.00001;
   // Acceptable float difference (for non-binarized BN outputs)
   float                         delta = 0.0001;
   int                           signLoc;
   // Number of iterations
   int                           iters = 20;
   // Number of mismatches
   int                           fail = 0;
   int                           misCnt = 0;
   int                           result = 0;
   // Layer size limits
   int                           fcMaxIn = 8192;
   int                           fcMaxOut = 8192;
   // Conv layer size limits
   int                           cnMaxActY = 256;
   int                           cnMaxActX = 256;
   int                           cnMaxActZ = 256;
   int                           cnMaxKrnY = 11;
   int                           cnMaxKrnX = 11;
   int                           cnMaxKnum = 512;
   int                           cnMaxPad  = 3;


   // Parse arguments
   while ((opt = getopt(argc, argv, "fcbs:pni:dl:h")) != -1) {
      switch (opt) {
         // FC mode
         case 'f':
            mode = FC_TEST;
            break;
         // CN mode
         case 'c':
            mode = CN_TEST;
            break;
         // Number of iterations
         case 'i':
            iters = atoi(optarg);
            break;
         case 'b':
            bn_mode = BN_TRUE;
            break;
         case 'n':
            out_bin = NO_OUT_BIN;
            break;
         case 's':
            sparsity = SPARSE;
            pruned = atoi(optarg);
            break;
         case 'p':
            packing = PACK;
            break;
         case 'd':
            pad = PAD;
            //padding = atoi(optarg);
            break;
         case 'l':
            pooling = POOL;
            pool = atoi(optarg);
            break;
         case 'h':
            printf("validation.c - tests XNOR/3PXNet implementations against reference\n");
            printf("Layer sizes are chosen randomly");
            printf("Options:\n");
            printf("-c : Concolutional layer\n");
            printf("-f : FC layer\n");
            printf("-i # : Number of iterations\n");
            printf("-b : Batch Norm\n");
            printf("-n : No output binarization\n");
            printf("-s # : Sparsity percentage(3PXNet)\n");
            printf("-p : Packed\n");
            printf("-d # : Padding size\n");
            printf("-l # : Pooling window size\n");
            exit(EXIT_SUCCESS);
         default:
            printf("Missing mode argument\n");
            exit(EXIT_FAILURE);
      }
   }

   // Intialize seed
   srand(0);

   // Open logfile
   time (&rawtime);
   strftime(timestr, sizeof(timestr), "_%y_%m_%d_%H_%M", localtime(&rawtime));
   if (mode == FC_TEST) {
      strcat(filename, "./log/fc");
   }
   else {
      strcat(filename, "./log/cn");
   }
#ifdef NEON
   strcat(filename, "_NEON");
#endif
   strcat(filename, timestr); 
   logFile = fopen(filename, "w+");
   if (mode == FC_TEST) {
      printf("Starting FC validation run at %s\n", asctime(localtime(&rawtime)));
      fprintf(logFile, "Starting FC validation run at %s\n", asctime(localtime(&rawtime)));

      // Check if batch norm is enabled
      if (bn_mode == BN_TRUE) {
         printf("Batch Normalization: Yes\n");
         fprintf(logFile, "Batch Normalization: Yes\n");
      }
      else {
         printf("Batch Normalization: No\n");
         fprintf(logFile, "Batch Normalization: No\n");
      }
      // Check if outputs are binarized 
      if (out_bin == OUT_BIN) {
         printf("Output Binarization: Yes\n");
         fprintf(logFile, "Output Binarization: Yes\n");
      }
      else {
         printf("Output Binarization: No\n");
         fprintf(logFile, "Output Binarization: No\n");
      }
      // Check if model is sparse 
      if (sparsity == SPARSE) {
         printf("Sparse: Yes\n");
         fprintf(logFile, "Sparse: Yes\n");
      }
      else {
         printf("Sparse: No\n");
         fprintf(logFile, "Sparse: No\n");
      }
      // Check if model is packed (only for sparse)
      if (packing == PACK) {
         printf("Packing: Yes\n");
         fprintf(logFile, "Packing: Yes\n");
      }
      else {
         printf("Packing: No\n");
         fprintf(logFile, "Packing: No\n");
      }
 

      for (int it = 0; it < iters; it++) {

         fprintf(logFile,"Running iteration %3d/%3d:\t", it, iters);
         printf("Running iteration %3d/%3d:\t", it, iters);

         // Generate parameters
         int fcIn = rand() % fcMaxIn;
         int fcOut = rand() % fcMaxOut;

         // Make sure minimum size is preserved
         if (fcIn  < pckWdt) { fcIn  = pckWdt; }
         if (fcOut < pckWdt) { fcOut = pckWdt; }

         // Constrain to multiple of pack width
         fcIn = fcIn/pckWdt * pckWdt;
         fcOut = fcOut/pckWdt * pckWdt;

         // Sparsity - number of active weights (per kernel)
         int actIn = (int)fcIn*(1.0 - (float)pruned/100.0);
         if (actIn < pckWdt) { actIn = pckWdt; }
         actIn = actIn/pckWdt * pckWdt;

         fprintf(logFile," In: %4d, Out: %4d, Sparsity: %4d%% (%4d/%4d)\t", fcIn, fcOut, pruned, actIn, fcIn);
         printf(" In: %4d, Out: %4d, Sparsity: %4d%% (%4d/%4d)\t", fcIn, fcOut, pruned, actIn, fcIn);

         // Reset mismatch count
         misCnt = 0;

         //////////////////////////////
         // Data Buffers
         // Activations
         int16_t     *actInt = malloc(fcIn*sizeof(int16_t));
         pckDtype    *actBin = malloc((fcIn/pckWdt)*sizeof(pckDtype));
         // Weights
         int16_t     *wgtRef = malloc(fcIn*fcOut*sizeof(int16_t));
         // Indices/weights for sparsity
         uint16_t    *indUc;
         uint8_t     *indPck;
         int16_t     *wgtImp;
         pckDtype    *wgtBin;
         if (sparsity == SPARSE) {
            // Weight memory is constant for both types of sparsity
            wgtImp = malloc(fcOut*(actIn)*sizeof(int16_t));
            wgtBin = malloc((fcOut*actIn/pckWdt)*sizeof(pckDtype));
            if (packing == PACK) {
               indPck = malloc(fcOut*(actIn/pckWdt)*sizeof(uint8_t));
               indUc  = NULL;
            }
            else {
               indUc  = malloc(fcOut*(actIn)*sizeof(uint16_t));
               indPck = NULL;
            }
         }
         else {
            wgtImp = malloc(fcOut*fcIn*sizeof(int16_t));
            wgtBin = malloc((fcOut*fcIn/pckWdt)*sizeof(pckDtype));
            indUc  = NULL;
            indPck = NULL;
         }
         // Outputs
         float       *outFlt = malloc(fcOut*sizeof(float));
         pckDtype    *outRef = malloc((fcOut/pckWdt)*sizeof(pckDtype));
         float       *outRfF = malloc(fcOut*sizeof(float));
         pckDtype    *outBin = malloc((fcOut/pckWdt)*sizeof(pckDtype));
         // Batch normalization
         float       *mean;
         float       *var;
         float       *gamma;
         float       *beta;
         float       *thresh;
         pckDtype    *sign;
         // Allocate BatchNorm buffers
         if (bn_mode == BN_TRUE) {
            mean   =  malloc(fcOut*sizeof(float));
            var    =  malloc(fcOut*sizeof(float));
            gamma  =  malloc(fcOut*sizeof(float));
            beta   =  malloc(fcOut*sizeof(float));
            thresh =  malloc(fcOut*sizeof(float));
            sign   =  malloc((fcOut/pckWdt)*sizeof(pckDtype));
         }
         // No BatchNorm, clear pointers
         else {
             mean     = NULL;
             var      = NULL;
             gamma    = NULL;
             beta     = NULL;
             thresh   = NULL;
             sign     = NULL;
         }



         //////////////////////////////
         // Randomize Data
         // Activations
         for (int actCnt = 0; actCnt < fcIn; actCnt++) {
            actInt[actCnt] = xnorVals[rand() % 2];  
         }
         // Sparse indices
         if (sparsity == SPARSE) {
            // Packed Sparse
            if (packing == PACK) {
               // For every output
               for (int outCnt = 0; outCnt<fcOut; outCnt++) {
                  // Generate indices
                  int low = 0; // Lowest possible index
                  int high = fcIn/pckWdt - actIn/pckWdt ; // Highest possible index
                  for (int indCnt = 0; indCnt < actIn/pckWdt; indCnt++) {
                     indPck[outCnt*actIn/pckWdt + indCnt] = (rand() % (high + 1 - low) + low);
                     // Adjust min/max
                     low = indPck[outCnt*actIn/pckWdt + indCnt] + 1;
                     high = fcIn/pckWdt - actIn/pckWdt + indCnt + 1;
                  }
               }
            }
            // Unconstrained Sparse
            else {
               // For every output
               for (int outCnt = 0; outCnt<fcOut; outCnt++) {
                  // Generate indices
                  int low = 0; // Lowest possible index
                  int high = fcIn - actIn ; // Highest possible index
                  for (int indCnt = 0; indCnt < actIn; indCnt++) {
                     indUc[outCnt*actIn + indCnt] = (rand() % (high + 1 - low) + low);
                     // Adjust min/max
                     low = indUc[outCnt*actIn + indCnt] + 1;
                     high = fcIn - actIn + indCnt + 1;
                  }
               }

            }
         }
         // Weights
         // Dense
         if (sparsity == DENSE) {
            for (int wgtCnt = 0; wgtCnt < fcIn*fcOut; wgtCnt++) {
               wgtRef[wgtCnt] = xnorVals[rand() % 2];  
               wgtImp[wgtCnt] = wgtRef[wgtCnt];  
            }
         }
         // Sparse
         else {
            // Zero them out beforehand
            for (int wgtCnt = 0; wgtCnt < fcIn*fcOut; wgtCnt++) {
               wgtRef[wgtCnt] = 0;            
            }
            // Packed
            if (packing == PACK) {
               // For reference only set active weights
               for (int wgtBkCnt = 0 ; wgtBkCnt < fcOut*actIn/pckWdt; wgtBkCnt++) {
                  for (int wgtCnt = 0; wgtCnt < pckWdt; wgtCnt++) {
                     // Starting with the non-zero index, fill out the next 32/64 weights
                     wgtRef[wgtBkCnt/(actIn/pckWdt)*fcIn + indPck[wgtBkCnt]*pckWdt + wgtCnt] = xnorVals[rand() % 2];
                     wgtImp[wgtBkCnt*pckWdt + wgtCnt] = wgtRef[wgtBkCnt/(actIn/pckWdt)*fcIn + indPck[wgtBkCnt]*pckWdt + wgtCnt];
                  }
               }
            }
            // Unconstrained
            else {
               // For reference only set active weights
               for (int wgtBkCnt = 0 ; wgtBkCnt < fcOut; wgtBkCnt++) {
                  for (int wgtCnt = 0 ; wgtCnt < actIn; wgtCnt++) {
                     wgtRef[wgtBkCnt*fcIn + indUc[wgtBkCnt*actIn + wgtCnt]] = xnorVals[rand() % 2];
                     wgtImp[wgtBkCnt*actIn + wgtCnt] = wgtRef[wgtBkCnt*fcIn + indUc[wgtBkCnt*actIn + wgtCnt]];
                  }
               }
            }
         }
         // Batch Normalization data
         if (bn_mode == BN_TRUE) {
            for (int bnCnt = 0; bnCnt < fcOut; bnCnt++) {
               mean[bnCnt]  = ((rand()/(float)RAND_MAX) * 2*fcIn) - fcIn;
               var[bnCnt]   = pow(((rand()/(float)RAND_MAX) * 2*fcIn) - fcIn, 2);
               gamma[bnCnt] = ((rand()/(float)RAND_MAX) * 1000.0 - 500.0);
               beta[bnCnt]  = ((rand()/(float)RAND_MAX) * 1000.0 - 500.0);
               // Calculate threshold based on other values
               // var is considered to be adjusted for epsilon and square rooted
               //thresh[bnCnt] = mean[bnCnt] - (beta[bnCnt]/gamma[bnCnt])*sqrt(var[bnCnt]+epsilon);
               thresh[bnCnt] = mean[bnCnt] - (beta[bnCnt]/gamma[bnCnt])*var[bnCnt];
               // Get sign and pack into binary vectors
               // Clear on the beginning of every pack
               if (bnCnt % pckWdt == 0) {
                  sign[bnCnt/pckWdt] = 0;
               }
               signLoc = gamma[bnCnt] >= 0;
               sign[bnCnt/pckWdt] |= signLoc << (pckWdt-1 - (bnCnt%pckWdt));
            }
         }
         

         //////////////////////////////
         // Run reference
         refFc(actInt, wgtRef, fcIn, fcOut, outFlt, mean, var, gamma, beta, epsilon);
         // Binarize outputs
         if (out_bin == OUT_BIN) {
            packBinThrsPtrFlt(outFlt, outRef, fcOut, 0);
         }
         // Binarize inputs
         packBinThrsPtr(actInt, actBin, fcIn, 0);
         // Binarize the weights
         if (sparsity == DENSE) {
            packBinThrsPtr(wgtImp, wgtBin, fcIn*fcOut, 0);
         }
         else {
            packBinThrsPtr(wgtImp, wgtBin, fcOut*actIn, 0);
         }
         // Run XNOR
         // Output Binarization
         if (out_bin == OUT_BIN) {
            // Dense
            if (sparsity == DENSE) {
               result = FcXnorWrap(actBin, wgtBin, fcIn, fcOut, outBin, thresh, sign);
            }
            // Sparse
            else {
               // Packed
               if (packing == PACK) {
                  result = Fc3pxnWrap(actBin, wgtBin, indPck, actIn, fcOut, outBin, thresh, sign);
               }
               // Unconstrained
               else {
                  fprintf(logFile,"Unconstrained sparse implementation currently not included");
                  result = 1;
               }
            }
         }
         else {
            if (sparsity == DENSE) {
               result = FcXnorNoBinWrap(actBin, wgtBin, fcIn, fcOut, outRfF, mean, var, gamma, beta);
            }
            // Sparse
            else {
               // Packed
               if (packing == PACK) {
                  result = Fc3pxnNoBinWrap(actBin, wgtBin, indPck, actIn, fcOut, outRfF, mean, var, gamma, beta);
               }
               // Unconstrained
               else {
                  fprintf(logFile,"Unconstrained sparse implementation currently not included");
                  result = 1;
               }

            }
         }

         //////////////////////////////
         // Run comparison
         if (out_bin == OUT_BIN) {
            for (int outBk = 0; outBk < fcOut/pckWdt; outBk++) {
               if (outRef[outBk] != outBin[outBk]) {
                  fprintf(logFile,"Mismatch in block %d: Ref: %X, Impl: %X\n", outBk, outRef[outBk], outBin[outBk]); 
                  misCnt++;
                  fail = 1;
               }
            }
         }
         else { 
            for (int outBk = 0; outBk < fcOut; outBk++) {
               if (fabs(outFlt[outBk] - outRfF[outBk]) > delta ) {
                  fprintf(logFile,"Mismatch in output %d: Ref: %f, Impl: %f\n", outBk, outFlt[outBk], outRfF[outBk]); 
                  misCnt++;
                  fail = 1;
               }
            }
         }

         if (misCnt || result) {
            fprintf(logFile, "FAIL\n");
            printf("FAIL\n");
         }
         else {
            fprintf(logFile, "PASS\n");
            printf("PASS\n");
         } 

         //////////////////////////////
         // Free memory
         free(actInt);
         free(actBin);
         free(wgtRef);
         free(wgtImp);
         free(wgtBin);
         free(outFlt);
         free(outRef);
         free(outRfF);
         free(outBin);
         // Batch normalization
         if (bn_mode == BN_TRUE) {
            free(mean);
            free(var);
            free(gamma);
            free(beta);
            free(thresh);
            free(sign);
         }
         // Sparse indices
         if (sparsity == SPARSE) {
            if (packing == PACK) {
               free(indPck);
            }
            else { // UNCONST
               free(indUc);
            }
         }
      }


   }
   else {
      printf("Starting CN validation run at %s\n", asctime(localtime(&rawtime)));
      fprintf(logFile, "Starting CN validation run at %s\n", asctime(localtime(&rawtime)));

      // Check if batch norm is enabled
      if (bn_mode == BN_TRUE) {
         printf("Batch Normalization: Yes\n");
         fprintf(logFile, "Batch Normalization: Yes\n");
      }
      else {
         printf("Batch Normalization: No\n");
         fprintf(logFile, "Batch Normalization: No\n");
      }
      // Check if outputs are binarized 
      if (out_bin == OUT_BIN) {
         printf("Output Binarization: Yes\n");
         fprintf(logFile, "Output Binarization: Yes\n");
      }
      else {
         printf("Output Binarization: No\n");
         fprintf(logFile, "Output Binarization: No\n");
      }
      // Check if model is sparse 
      if (sparsity == SPARSE) {
         printf("Sparse: Yes\n");
         fprintf(logFile, "Sparse: Yes\n");
      }
      else {
         printf("Sparse: No\n");
         fprintf(logFile, "Sparse: No\n");
      }
      // Check if model is packed (only for sparse)
      if (packing == PACK) {
         printf("Packing: Yes\n");
         fprintf(logFile, "Packing: Yes\n");
      }
      else {
         printf("Packing: No\n");
         fprintf(logFile, "Packing: No\n");
      }
      // Check if padding is used 
      if (pad == PAD) {
         printf("Padding: Yes\n");
         fprintf(logFile, "Padding: Yes\n");
      }
      else {
         printf("Padding: No\n");
         fprintf(logFile, "Padding: No\n");
      }
      // Check if pooling is used 
      if (pooling == POOL) {
         printf("Pooling: Yes\n");
         fprintf(logFile, "Pooling: Yes\n");
      }
      else {
         printf("Pooling: No\n");
         fprintf(logFile, "Pooling: No\n");
      }


      for (int it = 0; it < iters; it++) {

         fprintf(logFile,"Running iteration %3d/%3d:\t", it, iters);
         printf("Running iteration %3d/%3d:\t", it, iters);

         // Generate parameters
         int ActY = rand() % cnMaxActY;
         // For now keep activation square
         //int ActX = rand() % cnMaxActX;
         int ActX = ActY;
         int ActZ = rand() % cnMaxActZ;
         int KrnY = rand() % cnMaxKrnY;
         // For now keep activation square
         //int KrnX = rand() % cnMaxKrnX;
         int KrnX = KrnY;
         int Knum = rand() % cnMaxKnum;
         // Padding
         if (pad == PAD) {
            padding = rand() % cnMaxPad;
         }

         // Make sure nothing is zero
         if (KrnX <= padding) {KrnX = padding+1;}
         if (KrnY <= padding) {KrnY = padding+1;}
         if (ActX == 0) {ActX = 1;}
         if (ActY == 0) {ActY = 1;}
         // Make sure minimum size is preserved
         // Inputs cannot be smaller than kernels
         if ( ActY < KrnY)   { ActY = KrnY+1; }
         if ( ActX < KrnX)   { ActX = KrnX+1; }
         // Depth must be a multiple of pack width 
         if (ActZ < pckWdt) { ActZ = pckWdt; }
         // Output depth must be a multiple of pack width
         if (Knum < pckWdt) { Knum = pckWdt; }

         // Constrain depth to multiple of pack width
         ActZ = ActZ/pckWdt * pckWdt;
         Knum = Knum/pckWdt * pckWdt;

         // Make sure the overall kernel size makes it possible to use 8-bit indices
         // This is very basic, just clamp it down to lowest possible value
         // This could be more elaborate
         if (KrnY * KrnX * ActZ / pckWdt > 256) {
            ActZ = pckWdt;
         }

         // Make sure there's no truncation when pooling
         // Basically (ActY - KrnY + 2*padding + 1)/pool has to be an interger
         // Same for X
         if (pad == PAD && pooling == POOL && (ActY - KrnY + 2*padding + 1)%pool != 0) {
            // Regenerate padding and kernel values
            while(1) {
               padding = rand() % cnMaxPad;
               KrnY = rand() % cnMaxKrnY;
               KrnX = KrnY;
               if (KrnX <= padding) {KrnX = padding+1;}
               if (KrnY <= padding) {KrnY = padding+1;}
               if ((ActY - KrnY + 2*padding + 1)%pool == 0) {
               if ( ActY < KrnY)   { ActY = KrnY; }
               if ( ActX < KrnX)   { ActX = KrnX; }
                  break;
               }
            }
         }

         // For testing keep it small
         //ActY = 9  ; ActX = 9  ; ActZ = 32  ; KrnY = 3; KrnX = 3 ; Knum = 32 ; padding = 2;
         //padding = 3;

         // Sparsity - number of active weights (per kernel)
         int actIn = (int)(KrnY*KrnX*ActZ)*(1.0 - (float)pruned/100.0);
         if (actIn < pckWdt) { actIn = pckWdt; }
         actIn = actIn/pckWdt * pckWdt;

         fprintf(logFile," Act: %3dx%3dx%3d, Kernel: %2dx%2dx%3dx%3d, Padding: %d, Pooling: %d, Sparsity: %4d%% (%4d/%4d)\t", ActY, ActX, ActZ, KrnY, KrnX, ActZ, Knum, padding, pool, pruned, actIn, KrnY*KrnX*ActZ);
         printf(" Act: %3dx%3dx%3d, Kernel: %2dx%2dx%3dx%3d, Padding: %d, Pooling: %d, Sparsity: %4d%% (%4d/%4d)\t", ActY, ActX, ActZ, KrnY, KrnX, ActZ, Knum, padding, pool, pruned, actIn, KrnY*KrnX*ActZ);

         // Reset mismatch count
         misCnt = 0;

         //////////////////////////////
         // Data Buffers
         // Activations
         int16_t     *actInt = malloc(ActY*ActX*ActZ*sizeof(int16_t));
         int16_t     *actIntPd = malloc((ActY+2*padding)*(ActX+2*padding)*ActZ*sizeof(int16_t));
         pckDtype    *actBin = malloc((ActY*ActX*ActZ/pckWdt)*sizeof(pckDtype));
         // Weights
         int16_t     *wgtRef = malloc(KrnY*KrnX*ActZ*Knum*sizeof(int16_t));
         // Indices/weights for sparsity
         uint16_t    *indUc;
         uint8_t     *indPck;
         int16_t     *wgtImp;
         pckDtype    *wgtBin;
         if (sparsity == SPARSE) {
            // Weight memory is constant for both types of sparsity
            wgtImp = malloc(Knum*actIn*sizeof(int16_t));
            wgtBin = malloc((Knum*actIn/pckWdt)*sizeof(pckDtype));
            if (packing == PACK) {
               indPck = malloc(Knum*(actIn/pckWdt)*sizeof(uint8_t));
               indUc  = NULL;
            }
            else {
               indUc  = malloc(Knum*(actIn)*sizeof(uint16_t));
               indPck = NULL;
            }
         }
         else {
            wgtImp = malloc(KrnY*KrnX*ActZ*Knum*sizeof(int16_t));
            wgtBin = malloc((KrnY*KrnX*ActZ*Knum/pckWdt)*sizeof(pckDtype));
            indUc  = NULL;
            indPck = NULL;
         }
         // Outputs
         float       *outFlt = malloc(((ActY-KrnY+2*padding+1)/pool)*((ActX-KrnX+2*padding+1)/pool)*Knum*sizeof(float));
         pckDtype    *outRef = malloc((((ActY-KrnY+2*padding+1)/pool)*((ActX-KrnX+2*padding+1)/pool)*Knum/pckWdt)*sizeof(pckDtype));
         float       *outRfF = malloc(((ActY-KrnY+2*padding+1)/pool)*((ActX-KrnX+2*padding+1)/pool)*Knum*sizeof(float));
         pckDtype    *outBin = malloc((((ActY-KrnY+2*padding+1)/pool)*((ActX-KrnX+2*padding+1)/pool)*Knum/pckWdt)*sizeof(pckDtype));
         // Batch normalization
         float       *mean;
         float       *var;
         float       *gamma;
         float       *beta;
         float       *thresh;
         pckDtype    *sign;
         // Allocate BatchNorm buffers
         if (bn_mode == BN_TRUE) {
            mean   =  malloc(Knum*sizeof(float));
            var    =  malloc(Knum*sizeof(float));
            gamma  =  malloc(Knum*sizeof(float));
            beta   =  malloc(Knum*sizeof(float));
            thresh =  malloc(Knum*sizeof(float));
            sign   =  malloc((Knum/pckWdt)*sizeof(pckDtype));
         }
         // No BatchNorm, clear pointers
         else {
             mean     = NULL;
             var      = NULL;
             gamma    = NULL;
             beta     = NULL;
             thresh   = NULL;
             sign     = NULL;
         }



         //////////////////////////////
         // Randomize Data
         // Activations
         // Clear out padded data
         for (int actCnt = 0; actCnt < (ActY+2*padding)*(ActX+2*padding)*ActZ; actCnt++) {
            actIntPd[actCnt] = 0;
         }
         // Intialize non-padded data
         for (int actCnt = 0; actCnt < ActY*ActX*ActZ; actCnt++) {
            actInt[actCnt] = xnorVals[rand() % 2];  
            // first X padded rows padding*(ActX+2*padding)*ActZ
            // row offset paddingg y*padding+padding = actCnt/(ActX*ActZ)*2*padding*ActZ + padding*ActZ
            actIntPd[padding*(ActX+2*padding)*ActZ + (actCnt/(ActX*ActZ))*2*padding*ActZ + padding*ActZ + (actCnt/(ActX*ActZ))*ActX*ActZ + actCnt%(ActX*ActZ)] = actInt[actCnt];
         }
         // Sparse indices
         if (sparsity == SPARSE) {
            // Packed Sparse
            if (packing == PACK) {
               // For every kernel 
               for (int outCnt = 0; outCnt<Knum; outCnt++) {
                  // Generate indices
                  int low = 0; // Lowest possible index
                  int high = (KrnY*KrnX*ActZ)/pckWdt - actIn/pckWdt ; // Highest possible index
                  for (int indCnt = 0; indCnt < actIn/pckWdt; indCnt++) {
                     indPck[outCnt*actIn/pckWdt + indCnt] = (rand() % (high + 1 - low) + low);
                     // Adjust min/max
                     low = indPck[outCnt*actIn/pckWdt + indCnt] + 1;
                     high = (KrnY*KrnX*ActZ)/pckWdt - actIn/pckWdt + indCnt + 1;
                  }
               }
            }
            // Unconstrained Sparse
            else {
               // For every output
               for (int outCnt = 0; outCnt<Knum; outCnt++) {
                  // Generate indices
                  int low = 0; // Lowest possible index
                  int high = (KrnY*KrnX*ActZ) - actIn ; // Highest possible index
                  for (int indCnt = 0; indCnt < actIn; indCnt++) {
                     indUc[outCnt*actIn + indCnt] = (rand() % (high + 1 - low) + low);
                     // Adjust min/max
                     low = indUc[outCnt*actIn + indCnt] + 1;
                     high = (KrnY*KrnX*ActZ) - actIn + indCnt + 1;
                  }
               }

            }
         }
         // Weights
         // Dense
         if (sparsity == DENSE) {
            for (int wgtCnt = 0; wgtCnt < KrnY*KrnX*ActZ*Knum; wgtCnt++) {
               wgtRef[wgtCnt] = xnorVals[rand() % 2];  
               wgtImp[wgtCnt] = wgtRef[wgtCnt];  
            }
         }
         // Sparse
         else {
            // Zero them out beforehand
            for (int wgtCnt = 0; wgtCnt < KrnY*KrnX*ActZ*Knum; wgtCnt++) {
               wgtRef[wgtCnt] = 0;            
            }
            // Packed
            if (packing == PACK) {
               // For reference only set active weights
               for (int wgtBkCnt = 0 ; wgtBkCnt < Knum*actIn/pckWdt; wgtBkCnt++) {
                  for (int wgtCnt = 0; wgtCnt < pckWdt; wgtCnt++) {
                     // Starting with the non-zero index, fill out the next 32/64 weights
                     wgtRef[wgtBkCnt/(actIn/pckWdt)*(KrnY*KrnX*ActZ) + indPck[wgtBkCnt]*pckWdt + wgtCnt] = xnorVals[rand() % 2];
                     wgtImp[wgtBkCnt*pckWdt + wgtCnt] = wgtRef[wgtBkCnt/(actIn/pckWdt)*(KrnY*KrnX*ActZ) + indPck[wgtBkCnt]*pckWdt + wgtCnt];
                  }
               }
            }
            // Unconstrained
            else {
               // For reference only set active weights
               for (int wgtBkCnt = 0 ; wgtBkCnt < Knum; wgtBkCnt++) {
                  for (int wgtCnt = 0 ; wgtCnt < actIn; wgtCnt++) {
                     wgtRef[wgtBkCnt*KrnY*KrnX*ActZ + indUc[wgtBkCnt*actIn + wgtCnt]] = xnorVals[rand() % 2];
                     wgtImp[wgtBkCnt*actIn + wgtCnt] = wgtRef[wgtBkCnt*KrnX*KrnY*ActZ + indUc[wgtBkCnt*actIn + wgtCnt]];
                  }
               }
            }
         }
         // Batch Normalization data
         if (bn_mode == BN_TRUE) {
            for (int bnCnt = 0; bnCnt < Knum; bnCnt++) {
               mean[bnCnt]  = ((rand()/(float)RAND_MAX) * 2*KrnY*KrnX*ActZ) - KrnY*KrnX*ActZ;
               var[bnCnt]   = pow(((rand()/(float)RAND_MAX) * 2*KrnY*KrnX*ActZ) - KrnY*KrnX*ActZ, 2);
               gamma[bnCnt] = ((rand()/(float)RAND_MAX) * 1000.0 - 500.0);
               beta[bnCnt]  = ((rand()/(float)RAND_MAX) * 1000.0 - 500.0);
               // Calculate threshold based on other values
               thresh[bnCnt] = mean[bnCnt] - (beta[bnCnt]/gamma[bnCnt])*sqrt(var[bnCnt]+epsilon);
               // Get sign and pack into binary vectors
               // Clear on the beginning of every pack
               if (bnCnt % pckWdt == 0) {
                  sign[bnCnt/pckWdt] = 0;
               }
               signLoc = gamma[bnCnt] >= 0;
               sign[bnCnt/pckWdt] |= signLoc << (pckWdt-1 - (bnCnt%pckWdt));
            }
         }
         

         ////////////////////////////////
         //// Run reference
         refCn(actIntPd, wgtRef, ActZ, (ActX+2*padding), (ActY+2*padding), KrnX, KrnY, Knum, outFlt, pool, mean, var, gamma, beta, epsilon);
         // Binarize outputs
         if (out_bin == OUT_BIN) {
            packBinThrsPtrFlt(outFlt, outRef, ((ActY+2*padding-KrnY+1)/pool)*((ActX+2*padding-KrnX+1)/pool)*Knum, 0);
         }
         // Binarize inputs
         packBinThrsPtr(actInt, actBin, ActY*ActX*ActZ, 0);
         // Binarize the weights
         if (sparsity == DENSE) {
            packBinThrsPtr(wgtImp, wgtBin, KrnY*KrnX*ActZ*Knum, 0);
         }
         else {
            packBinThrsPtr(wgtImp, wgtBin, Knum*actIn, 0);
         }
         // Run XNOR
         // Output Binarization
         if (out_bin == OUT_BIN) {
            // Dense
            if (sparsity == DENSE) {
               result = CnXnorWrap(actBin, wgtBin, ActZ, ActX, ActY, ActZ, KrnX, KrnY, Knum, outBin, padding, pool, thresh, sign);
            }
            // Sparse
            else {
               // Packed
               if (packing == PACK) {
                  result = Cn3pxnWrap(actBin, wgtBin, indPck, actIn, ActZ, ActX, ActY, ActZ, KrnX, KrnY, Knum, outBin, padding, pool, thresh, sign);
               }
               // Unconstrained
               //else {
               //   result = xnorFCSprUc(actBin, wgtBin, indUc, actIn, fcOut, outBin, thresh, sign);
               //}
            }
         }
         //else {
         //   if (sparsity == DENSE) {
         //      result = xnorFCDnsNoBin(actBin, wgtBin, fcIn, fcOut, outRfF, mean, var, gamma, beta, epsilon);
         //   }
         //   // Sparse
         //   else {
         //      // Packed
         //      if (packing == PACK) {
         //         result = xnorFCSprPckNoBin(actBin, wgtBin, indPck, actIn, fcOut, outRfF, mean, var, gamma, beta, epsilon);
         //      }
         //      // Unconstrained
         //      else {
         //         result = xnorFCSprUcNoBin(actBin, wgtBin, indUc, actIn, fcOut, outRfF, mean, var, gamma, beta, epsilon);
         //      }

         //   }
         //}

         //////////////////////////////
         // Run comparison
         if (out_bin == OUT_BIN) {
            for (int outBk = 0; outBk < ((ActY+2*padding-KrnY+1)/pool)*((ActX+2*padding-KrnX+1)/pool)*Knum/pckWdt; outBk++) {
               if (outRef[outBk] != outBin[outBk]) {
                  fprintf(logFile,"Mismatch in block %d: Ref: %X, Impl: %X\n", outBk, outRef[outBk], outBin[outBk]); 
                  misCnt++;
                  fail = 1;
               }
            }
         }
         //else { 
         //   for (int outBk = 0; outBk < fcOut; outBk++) {
         //      if (fabs(outFlt[outBk] - outRfF[outBk]) > delta ) {
         //         fprintf(logFile,"Mismatch in output %d: Ref: %f, Impl: %f\n", outBk, outFlt[outBk], outRfF[outBk]); 
         //         misCnt++;
         //         fail = 1;
         //      }
         //   }
         //}

         if (misCnt || result) {
            fprintf(logFile, "FAIL\n");
            printf("FAIL\n");
         }
         else {
            fprintf(logFile, "PASS\n");
            printf("PASS\n");
         } 

         //////////////////////////////////
         ////// Free memory
         free(actInt);
         free(actIntPd);
         free(actBin);
         free(wgtRef);
         free(wgtImp);
         free(wgtBin);
         free(outFlt);
         free(outRef);
         free(outRfF);
         free(outBin);
         //// Batch normalization
         if (bn_mode == BN_TRUE) {
            free(mean);
            free(var);
            free(gamma);
            free(beta);
            free(thresh);
            free(sign);
         }
         // Sparse indices
         if (sparsity == SPARSE) {
            if (packing == PACK) {
               free(indPck);
            }
            else { // UNCONST
               free(indUc);
            }
         }
      }


   }
  
   time (&rawtime);
   if (fail == 0) {
      printf("Validation run PASS %s",asctime(localtime(&rawtime)) );
      fprintf(logFile, "Validation run PASS %s",asctime(localtime(&rawtime)) );
      fclose(logFile);
      return (EXIT_SUCCESS);
   }
   else {
      printf("Validation run FAIL %s",asctime(localtime(&rawtime)) );
      fprintf(logFile, "Validation run FAIL %s",asctime(localtime(&rawtime)) );
      fclose(logFile);
      return (EXIT_FAILURE);
   }


}

