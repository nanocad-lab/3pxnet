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
 * \file      3pxnet_cn.c
 * \brief     3PXNet convolutional layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#include "3pxnet_cn.h"

/**
 * @details 3pxnet binarized convolutional (CN) layer with output binarization - general wrapper.
 * Selects the appropriate implementation (batch normalization, pooling, padding, NEON support)
 * 
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix 
 * @param[in] kLen - sparse kernel size 
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] pad  - padding size 
 * @param[in] pool - pooling window size 
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 * @return 0 - Success, 1 - Failure
 */

uint8_t Cn3pxnWrap(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint16_t pad, const uint16_t pool, bnDtype * __restrict thresh, pckDtype * sign) {

   // Batch Norm present (thresh != NULL)
   if (thresh) {
      // Output or input depth not multiple of pack width - not supported atm
      if (dpth  % pckWdt != 0 || knum % pckWdt != 0 ) {
         return 1;
      }
      // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
      else if (dpth  % (4*pckWdt) == 0) {
         return 1;
      }
      // Check for dual SIMD support
      else if (dpth  % (2*pckWdt) == 0) {
         return 1;
      }
#endif
#endif
      // Roll back to default implementation
      else {
         // Non-padded
         if (pad == 0) {
            // No pooling
            if (pool == 1) {
               CnBn3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, thresh, sign);
               return 0;
            }
            // Pooling
            else {
               // This should be a non-padded version
               CnBnPdPl3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool, thresh, sign);
               return 0;
            }
         }
         // Padded
         else {
            // No pooling
            if (pool == 1) {
               CnBnPd3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, thresh, sign);
               return 0;
            }
            // Pooling
            else {
               CnBnPdPl3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool, thresh, sign);
               return 0;
            }
         }
      }
   }
   // No Batch Norm (bnDtype == NULL)
   else {
      // Output or input not depth multiple of pack width - not supported atm
      if (dpth % pckWdt != 0 || knum % pckWdt != 0 ) {
         return 1;
      }
      // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
      else if (dpth  % (4*pckWdt) == 0) {
         return 1;
      }
      // Check for dual SIMD support
      else if (dpth  % (2*pckWdt) == 0) {
         return 1;
      }
#endif
#endif
      // Roll back to default implementation
      else {
         // No padding
         if (pad == 0) {
            // No pooling
            if (pool == 1) {
               Cn3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut);
            }
            // Pooling
            else {
               // This should be changed to non-padded implementations
               CnPdPl3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool);
            }
         }
         // Padding
         else {
            // No pooling
            if (pool == 1) {
               CnPd3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad);
            }
            // Pooling
            else {
               CnPdPl3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool);
            }
         }
         return 0;
      }
   }

}
/**
 * @details 3pxnet binarized convolutional (CN) layer with output binarization - general wrapper.
 * Selects the appropriate implementation (batch normalization, pooling, padding, NEON support)
 *
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix
 * @param[in] kLen - sparse kernel size
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] pad  - padding size
 * @param[in] pool - pooling window size
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 * @return 0 - Success, 1 - Failure
 */
uint8_t Cn3pxnNoBinWrap(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, uint8_t* __restrict pInd, 
    const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, 
    const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, 
    bnDtype* __restrict pOut, const uint16_t pad, const uint16_t pool, 
    bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {

    // Batch Norm present (thresh != NULL)
    if (mean) {
        // Output or input depth not multiple of pack width - not supported atm
        if (dpth % pckWdt != 0 ) {
            return 1;
        }
        // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
        else if (dpth % (4 * pckWdt) == 0) {
            return 1;
        }
        // Check for dual SIMD support
        else if (dpth % (2 * pckWdt) == 0) {
            return 1;
        }
#endif
#endif
        // Roll back to default implementation
        else {
            // Non-padded
            if (pad == 0) {
                // No pooling
                if (pool == 1) {
                    CnBn3pxnNoBin(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, mean,var,gamma,beta);
                    return 0;
                }
                // Pooling
                else {
                    // This should be a non-padded version
                    CnBnPdPl3pxnNoBin(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool, mean, var, gamma, beta);
                    return 0;
                }
            }
            // Padded
            else {
                // No pooling
                if (pool == 1) {
                    CnBnPd3pxnNoBin(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, mean, var, gamma, beta);
                    return 0;
                }
                // Pooling
                else {
                    CnBnPdPl3pxnNoBin(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool, mean, var, gamma, beta);
                    return 0;
                }
            }
        }
    }
    // No Batch Norm (bnDtype == NULL)
    /*else {
        // Output or input not depth multiple of pack width - not supported atm
        if (dpth % pckWdt != 0 || knum % pckWdt != 0) {
            return 1;
        }
        // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
        else if (dpth % (4 * pckWdt) == 0) {
            return 1;
        }
        // Check for dual SIMD support
        else if (dpth % (2 * pckWdt) == 0) {
            return 1;
        }
#endif
#endif
        // Roll back to default implementation
        else {
            // No padding
            if (pad == 0) {
                // No pooling
                if (pool == 1) {
                    Cn3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut);
                }
                // Pooling
                else {
                    // This should be changed to non-padded implementations
                    CnPdPl3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool);
                }
            }
            // Padding
            else {
                // No pooling
                if (pool == 1) {
                    CnPd3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad);
                }
                // Pooling
                else {
                    CnPdPl3pxn(pAct, pKrn, pInd, kLen, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool);
                }
            }
            return 0;
        }
    }
    */
}
/**
 * @details 3PXNet binarized convolutional (CN) layer with output binarization - array/no padding/ no pooling/.
 * Kernel as output loop dimension.
 * 
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix 
 * @param[in] kLen - sparse kernel size 
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector
 */
void Cn3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   uint8_t  *idxY = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxX = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxZ = malloc(kLen*sizeof(uint8_t));
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = kdpt/pckWdt;
   uint16_t index = 0;

   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Packed slices
      for (uint16_t ks = 0; ks<pckWdt; ks++) {
         // Loop through all indices and pack them locally
         for (uint8_t kll = 0; kll < kLen/pckWdt; kll++) {
            idxY[kll] = pInd[index + kll]/(kwdt*kCoeff);
            idxX[kll] = (pInd[index + kll]-idxY[kll]*kwdt*kCoeff)/kCoeff;
            idxZ[kll] = pInd[index + kll]-idxY[kll]*kwdt*kCoeff- idxX[kll]*kCoeff;
         }
         // Y dim
         for (uint16_t y = 0; y < (hght-khgt+1); y++) {
            // X dim
            for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
               // Need to do that because we'll be oring into it 
               if (ks == 0) {
                  pOut[y*(wdth-kwdt+1)*knum/pckWdt + x*knum/pckWdt + k] = 0;
               }
               outTemp = 0;
               for (uint16_t kl = 0; kl < kLen/pckWdt; kl++) { 
                  // XNOR multiplication
                  xnorTmp = ~( pAct[(y+idxY[kl])*yCoeff + (x+idxX[kl])*xCoeff + idxZ[kl]] ^ pKrn[index + kl]);
                  //
                  // popcount
                  xnorTmp = popcount(xnorTmp);
                  // Accumulation
                  outTemp += xnorTmp;
               } // K-len
               // Adjust the output value
               outTemp = outTemp - (kLen - outTemp);
               // Binarize
               outTemp = outTemp >= 0;
               // Shift based on current kernel slice
               outTemp = outTemp << (pckWdt-1-ks);
               // First time writing to a given word, make sure to clear it
               // Write out
               pOut[y*(wdth-kwdt+1)*knum/pckWdt + x*knum/pckWdt + k] |= outTemp;
            }
         }
         index += kLen/pckWdt;
      }
   }
}

/**
 * @details 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/ no pooling/.
 * Kernel as output loop dimension.
 * 
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix 
 * @param[in] kLen - sparse kernel size 
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] pad  - padding size
 */
void CnPd3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   uint8_t  *idxY = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxX = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxZ = malloc(kLen*sizeof(uint8_t));
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = kdpt/pckWdt;
   uint16_t index = 0;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   uint8_t  idxYY = 0;
   uint8_t  idxXX = 0;

   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Packed slices
      for (uint16_t ks = 0; ks<pckWdt; ks++) {
         // Loop through all indices and pack them locally
         for (uint16_t kll = 0; kll < kLen/pckWdt; kll++) {
            idxY[kll] = pInd[index + kll]/(kwdt*kCoeff);
            idxX[kll] = (pInd[index + kll]-idxY[kll]*kwdt*kCoeff)/kCoeff;
            idxZ[kll] = pInd[index + kll]-idxY[kll]*kwdt*kCoeff- idxX[kll]*kCoeff;
         }
         // Y dim
         for (uint16_t y = 0; y < (hght-khgt+2*pad+1); y++) {
            // X dim
            for (uint16_t x = 0; x < (wdth-kwdt+2*pad+1); x++) {
               // Need to do that because we'll be oring into it 
               if (ks == 0) {
                  pOut[y*(wdth-kwdt+2*pad+1)*knum/pckWdt + x*knum/pckWdt + k] = 0;
               }
               outTemp = 0;
               xyCount = 0;
               for (uint16_t kl = 0; kl < kLen/pckWdt; kl++) { 
                  idxYY = y+idxY[kl];
                  idxXX = x+idxX[kl];
                  if (!(idxYY < pad || idxXX < pad || idxYY-pad >= hght || idxXX-pad >= wdth)) {
                     xyCount++;
                     // XNOR multiplication
                     xnorTmp = ~( pAct[(idxYY-pad)*yCoeff + (idxXX-pad)*xCoeff + idxZ[kl]] ^ pKrn[(index + kl)]);
                     // popcount
                     xnorTmp = popcount(xnorTmp);
                     // Accumulation
                     outTemp += xnorTmp;
                  }
               } // K-len
               // Adjust the output value
               outTemp = outTemp - (xyCount*pckWdt - outTemp);
               // Binarize
               outTemp = outTemp >= 0;
               // Shift based on current kernel slice
               outTemp = outTemp << (pckWdt-1-ks);
               // First time writing to a given word, make sure to clear it
               // Write out
               pOut[y*(wdth-kwdt+2*pad+1)*knum/pckWdt + x*knum/pckWdt + k] |= outTemp;
            }
         }
         index += kLen/pckWdt;
      }
   }
}

/**
 * @details 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/pooling/.
 * Kernel as output loop dimension.
 * 
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix 
 * @param[in] kLen - sparse kernel size 
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] pad  - padding size
 * @param[in] pool - pooling window size
 */
void CnPdPl3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   uint8_t  *idxY = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxX = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxZ = malloc(kLen*sizeof(uint8_t));
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = kdpt/pckWdt;
   uint16_t index = 0;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   uint8_t  idxYY = 0;
   uint8_t  idxXX = 0;
   // For maxpooling
   int32_t  maxTemp = 0;

   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Packed slices
      for (uint16_t ks = 0; ks<pckWdt; ks++) {
         // Loop through all indices and pack them locally
         for (uint8_t kll = 0; kll < kLen/pckWdt; kll++) {
            idxY[kll] = pInd[index + kll]/(kwdt*kCoeff);
            idxX[kll] = (pInd[index + kll]-idxY[kll]*kwdt*kCoeff)/kCoeff;
            idxZ[kll] = pInd[index + kll]-idxY[kll]*kwdt*kCoeff- idxX[kll]*kCoeff;
         }
         // Y dim
         for (uint16_t y = 0; y < ((hght-khgt+2*pad+1)/pool); y++) {
            // X dim
            for (uint16_t x = 0; x < ((wdth-kwdt+2*pad+1)/pool); x++) {
               maxTemp = -(khgt*kwdt*kdpt);
               // Need to do that because we'll be oring into it 
               if (ks == 0) {
                  pOut[y*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt + x*knum/pckWdt + k] = 0;
               }
               for (uint16_t yy = 0; yy < pool; yy++) {
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     outTemp = 0;
                     xyCount = 0;
                     for (uint16_t kl = 0; kl < kLen/pckWdt; kl++) { 
                        idxYY = y*pool+yy+idxY[kl];
                        idxXX = x*pool+xx+idxX[kl];
                        if (!(idxYY < pad || idxXX < pad || idxYY-pad >= hght || idxXX-pad >= wdth)) {
                           xyCount++;
                           // XNOR multiplication
                           xnorTmp = ~( pAct[(idxYY-pad)*yCoeff + (idxXX-pad)*xCoeff + idxZ[kl]] ^ pKrn[(index + kl)]);
                           // popcount
                           xnorTmp = popcount(xnorTmp);
                           // Accumulation
                           outTemp += xnorTmp;
                        }
                     } // K-len
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*pckWdt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = maxTemp >= 0;
               // Shift based on current kernel slice
               maxTemp = maxTemp << (pckWdt-1-ks);
               // First time writing to a given word, make sure to clear it
               // Write out
               pOut[y*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt + x*knum/pckWdt + k] |= maxTemp;
               //goto end;
            }
         }
         index += kLen/pckWdt;
      }
   }
}

/**
 * @details 3PXNet binarized convolutional (CN) layer with output binarization - array/no padding/ no pooling/batch norm.
 * Kernel as output loop dimension. Batchnorm implemented as a threshold function.
 * 
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix 
 * @param[in] kLen - sparse kernel size 
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBn3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   uint8_t  *idxY = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxX = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxZ = malloc(kLen*sizeof(uint8_t));
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = kdpt/pckWdt;
   uint16_t index = 0;
   pckDtype  signs = 0;

   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Packed slices
      for (uint16_t ks = 0; ks<pckWdt; ks++) {
         // Loop through all indices and pack them locally
         for (uint8_t kll = 0; kll < kLen/pckWdt; kll++) {
            idxY[kll] = pInd[index + kll]/(kwdt*kCoeff);
            idxX[kll] = (pInd[index + kll]-idxY[kll]*kwdt*kCoeff)/kCoeff;
            idxZ[kll] = pInd[index + kll]-idxY[kll]*kwdt*kCoeff- idxX[kll]*kCoeff;
         }
         // Y dim
         for (uint16_t y = 0; y < (hght-khgt+1); y++) {
            // X dim
            for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
               // Need to do that because we'll be oring into it 
               if (ks == 0) {
                  pOut[y*(wdth-kwdt+1)*knum/pckWdt + x*knum/pckWdt + k] = 0;
               }
               outTemp = 0;
               for (uint16_t kl = 0; kl < kLen/pckWdt; kl++) { 
                  // XNOR multiplication
                  xnorTmp = ~( pAct[(y+idxY[kl])*yCoeff + (x+idxX[kl])*xCoeff + idxZ[kl]] ^ pKrn[index + kl]);
                  // popcount
                  xnorTmp = popcount(xnorTmp);
                  // Accumulation
                  outTemp += xnorTmp;
               } // K-len
               // Adjust the output value
               outTemp = outTemp - (kLen - outTemp);
               // Binarize
               outTemp = (bnPrec) outTemp >= *thresh;
               // Shift based on current kernel slice
               outTemp = outTemp << (pckWdt-1-ks);
               // First time writing to a given word, make sure to clear it
               // Write out
               pOut[y*(wdth-kwdt+1)*knum/pckWdt + x*knum/pckWdt + k] |= outTemp;
            }
         }
         index += kLen/pckWdt;
         thresh++;
      }
   }

   // Adjust signs
   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Grab a batch of bn signs for kernels
      signs = *sign++;
      // Y dim
      for (uint16_t y = 0; y < (hght-khgt+1); y++) {
         // X dim
         for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
            pOut[y*(wdth-kwdt+1)*knum/pckWdt + x*knum/pckWdt + k] = ~(pOut[y*(wdth-kwdt+1)*knum/pckWdt + x*knum/pckWdt + k] ^ signs);
         }
      }
   }

   free(idxY);
   free(idxX);
   free(idxZ);
}

void CnBn3pxnNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, uint8_t* __restrict pInd, 
    const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, 
    const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, 
    bnDtype* __restrict pOut, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {

    // Temporary variables
    pckDtype xnorTmp = 0;
    int32_t  outTemp = 0;
    uint8_t* idxY = malloc(kLen * sizeof(uint8_t));
    uint8_t* idxX = malloc(kLen * sizeof(uint8_t));
    uint8_t* idxZ = malloc(kLen * sizeof(uint8_t));
    uint16_t  yCoeff = wdth * dpth / pckWdt;
    uint16_t  xCoeff = dpth / pckWdt;
    uint16_t  kCoeff = kdpt / pckWdt;
    uint16_t index = 0;
    pckDtype  signs = 0;

    // Outer loop - kernels
    for (uint16_t k = 0; k < knum; k++) {
        // Loop through all indices and pack them locally
        for (uint8_t kll = 0; kll < kLen / pckWdt; kll++) {
            idxY[kll] = pInd[index + kll] / (kwdt * kCoeff);
            idxX[kll] = (pInd[index + kll] - idxY[kll] * kwdt * kCoeff) / kCoeff;
            idxZ[kll] = pInd[index + kll] - idxY[kll] * kwdt * kCoeff - idxX[kll] * kCoeff;
        }
        // Y dim
        for (uint16_t y = 0; y < (hght - khgt + 1); y++) {
            // X dim
            for (uint16_t x = 0; x < (wdth - kwdt + 1); x++) {
                // Need to do that because we'll be oring into it 
                outTemp = 0;
                for (uint16_t kl = 0; kl < kLen / pckWdt; kl++) {
                    // XNOR multiplication
                    xnorTmp = ~(pAct[(y + idxY[kl]) * yCoeff + (x + idxX[kl]) * xCoeff + idxZ[kl]] ^ pKrn[index + kl]);
                    // popcount
                    xnorTmp = popcount(xnorTmp);
                    // Accumulation
                    outTemp += xnorTmp;
                } // K-len
                // Adjust the output value
                outTemp = outTemp - (kLen - outTemp);
                // Binarize
                *pOut++ = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                // Shift based on current kernel slice
                /*outTemp = outTemp << (pckWdt - 1 - ks);
                // First time writing to a given word, make sure to clear it
                // Write out
                pOut[y * (wdth - kwdt + 1) * knum / pckWdt + x * knum / pckWdt + k] |= outTemp;*/
            }
        }
        index += kLen / pckWdt;
        //thresh++;
    }
    free(idxY);
    free(idxX);
    free(idxZ);
}

/**
 * @details 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/ no pooling/batch norm.
 * Kernel as output loop dimension. Batchnorm implemented as a threshold function.
 * 
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix 
 * @param[in] kLen - sparse kernel size 
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] pad  - padding size 
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPd3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   uint8_t  *idxY = malloc((kLen/pckWdt)*sizeof(uint8_t));
   uint8_t  *idxX = malloc((kLen/pckWdt)*sizeof(uint8_t));
   uint8_t  *idxZ = malloc((kLen/pckWdt)*sizeof(uint8_t));
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = kdpt/pckWdt;
   uint16_t index = 0;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   uint8_t  idxYY = 0;
   uint8_t  idxXX = 0;
   pckDtype  signs = 0;

   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Packed slices
      for (uint16_t ks = 0; ks<pckWdt; ks++) {
         // Loop through all indices and pack them locally
         for (uint16_t kll = 0; kll < kLen/pckWdt; kll++) {
            idxY[kll] = pInd[index + kll]/(kwdt*kCoeff);
            idxX[kll] = (pInd[index + kll]-idxY[kll]*kwdt*kCoeff)/kCoeff;
            idxZ[kll] = pInd[index + kll]-idxY[kll]*kwdt*kCoeff- idxX[kll]*kCoeff;
         }
         // Y dim
         for (uint16_t y = 0; y < (hght-khgt+2*pad+1); y++) {
            // X dim
            for (uint16_t x = 0; x < (wdth-kwdt+2*pad+1); x++) {
               // Need to do that because we'll be oring into it 
               if (ks == 0) {
                  pOut[y*(wdth-kwdt+2*pad+1)*knum/pckWdt + x*knum/pckWdt + k] = 0;
               }
               outTemp = 0;
               xyCount = 0;
               for (uint16_t kl = 0; kl < kLen/pckWdt; kl++) { 
                  idxYY = y+idxY[kl];
                  idxXX = x+idxX[kl];
                  //if (!(idxYY < pad || idxXX < pad || idxYY >= hght-khgt+2*pad+1 || idxXX >= wdth-kwdt+2*pad+1)) {
                  if (!(idxYY < pad || idxXX < pad || idxYY-pad >= hght || idxXX-pad >= wdth)) {
                     xyCount++;
                     // XNOR multiplication
                     xnorTmp = ~( pAct[(idxYY-pad)*yCoeff + (idxXX-pad)*xCoeff + idxZ[kl]] ^ pKrn[(index + kl)]);
                     // popcount
                     xnorTmp = popcount(xnorTmp);
                     // Accumulation
                     outTemp += xnorTmp;
                  }
               } // K-len
               // Adjust the output value
               outTemp = outTemp - (xyCount*pckWdt - outTemp);
               // Binarize
               outTemp = (bnPrec) outTemp >= *thresh;
               // Shift based on current kernel slice
               outTemp = outTemp << (pckWdt-1-ks);
               // First time writing to a given word, make sure to clear it
               // Write out
               pOut[y*(wdth-kwdt+2*pad+1)*knum/pckWdt + x*knum/pckWdt + k] |= outTemp;
            }
         }
         index += kLen/pckWdt;
         thresh++;
      }
   }
   // Adjust signs
   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Grab a batch of bn signs for kernels
      signs = *sign++;
      // Y dim
      for (uint16_t y = 0; y < ((hght-khgt+2*pad)+1); y++) {
         // X dim
         for (uint16_t x = 0; x < ((wdth-kwdt+2*pad)+1); x++) {
            pOut[y*(((wdth-kwdt+2*pad))+1)*knum/pckWdt + x*knum/pckWdt + k] = ~(pOut[y*(((wdth-kwdt+2*pad))+1)*knum/pckWdt + x*knum/pckWdt + k] ^ signs);
         }
      }
   }
   
   free(idxY);
   free(idxX);
   free(idxZ);

}

void CnBnPd3pxnNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, uint8_t* __restrict pInd, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {

    // Temporary variables
    pckDtype xnorTmp = 0;
    int32_t  outTemp = 0;
    uint8_t* idxY = malloc((kLen / pckWdt) * sizeof(uint8_t));
    uint8_t* idxX = malloc((kLen / pckWdt) * sizeof(uint8_t));
    uint8_t* idxZ = malloc((kLen / pckWdt) * sizeof(uint8_t));
    uint16_t  yCoeff = wdth * dpth / pckWdt;
    uint16_t  xCoeff = dpth / pckWdt;
    uint16_t  kCoeff = kdpt / pckWdt;
    uint16_t index = 0;
    // XY count for padding adjustment
    uint8_t  xyCount = 0;
    uint8_t  idxYY = 0;
    uint8_t  idxXX = 0;
    pckDtype  signs = 0;

    // Outer loop - kernels
    for (uint16_t k = 0; k < knum ; k++) {
            // Loop through all indices and pack them locally
        for (uint16_t kll = 0; kll < kLen / pckWdt; kll++) {
            idxY[kll] = pInd[index + kll] / (kwdt * kCoeff);
            idxX[kll] = (pInd[index + kll] - idxY[kll] * kwdt * kCoeff) / kCoeff;
            idxZ[kll] = pInd[index + kll] - idxY[kll] * kwdt * kCoeff - idxX[kll] * kCoeff;
        }
        // Y dim
        for (uint16_t y = 0; y < (hght - khgt + 2 * pad + 1); y++) {
            // X dim
            for (uint16_t x = 0; x < (wdth - kwdt + 2 * pad + 1); x++) {
                // Need to do that because we'll be oring into it 
                if (k == 0) {
                    pOut[y * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt + x * knum / pckWdt + k] = 0;
                }
                outTemp = 0;
                xyCount = 0;
                for (uint16_t kl = 0; kl < kLen / pckWdt; kl++) {
                    idxYY = y + idxY[kl];
                    idxXX = x + idxX[kl];
                    //if (!(idxYY < pad || idxXX < pad || idxYY >= hght-khgt+2*pad+1 || idxXX >= wdth-kwdt+2*pad+1)) {
                    if (!(idxYY < pad || idxXX < pad || idxYY - pad >= hght || idxXX - pad >= wdth)) {
                        xyCount++;
                        // XNOR multiplication
                        xnorTmp = ~(pAct[(idxYY - pad) * yCoeff + (idxXX - pad) * xCoeff + idxZ[kl]] ^ pKrn[(index + kl)]);
                        // popcount
                        xnorTmp = popcount(xnorTmp);
                        // Accumulation
                        outTemp += xnorTmp;
                    }
                } // K-len
                // Adjust the output value
                outTemp = outTemp - (xyCount * pckWdt - outTemp);
                // Binarize
                pOut[y * (wdth - kwdt + 1) * knum / pckWdt + x * knum / pckWdt + k] = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                // Shift based on current kernel slice
                //outTemp = outTemp << (pckWdt - 1 - ks);
                // First time writing to a given word, make sure to clear it
                // Write out
                //pOut[y * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt + x * knum / pckWdt + k] |= outTemp;
            }
        }
        index += kLen / pckWdt;
        //thresh++;
    }
    // Adjust signs
    // Outer loop - kernels

    free(idxY);
    free(idxX);
    free(idxZ);

}

#ifdef NEON
/**
 * @details 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/ no pooling/batch norm/NEON
 * Kernel as output loop dimension. Batchnorm implemented as a threshold function.
 * 
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix 
 * @param[in] kLen - sparse kernel size 
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] pad  - padding size 
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 */

void CnBnPd3pxnNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, const uint8_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   uint8_t  *idxY = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxX = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxZ = malloc(kLen*sizeof(uint8_t));
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = kdpt/pckWdt;
   uint16_t index = 0;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   uint8_t  idxYY = 0;
   uint8_t  idxXX = 0;
   pckDtype  signs = 0;
   int8_t    signCur  = 0;
   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int32x4_t mask;
   int64x2_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[4];

   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Packed slices
      for (uint16_t ks = 0; ks<pckWdt; ks++) {
         // Loop through all indices and pack them locally
         for (uint8_t kll = 0; kll < kLen; kll++) {
            idxY[kll] = pInd[index + kll]/(kwdt*kCoeff);
            idxX[kll] = (pInd[index + kll]-idxY[kll]*kwdt*kCoeff)/kCoeff;
            idxZ[kll] = pInd[index + kll]-idxY[kll]*kwdt*kCoeff- idxX[kll]*kCoeff;
         }
         // Y dim
         for (uint16_t y = 0; y < (hght-khgt+2*pad+1); y++) {
            // X dim
            for (uint16_t x = 0; x < (wdth-kwdt+2*pad+1); x++) {
               // Need to do that because we'll be oring into it 
               if (ks == 0) {
                  pOut[y*(wdth-kwdt+2*pad+1)*knum/pckWdt + x*knum/pckWdt + k] = 0;
               }
               xyCount = 0;
               vecOut[0] = 0;
               vecOut[1] = 0;
               for (uint8_t kl = 0; kl < kLen/4; kl++) { 
                  for (uint8_t kll = 0; kll < 4; kll++) {
                     // Check indices
                     idxYY = y+idxY[4*kl+kll];
                     idxXX = x+idxX[4*kl+kll];
                     // Load the values
                     if (!(idxYY < pad || idxXX < pad || idxYY > hght-khgt+2*pad+1 || idxXX > wdth-kwdt+2*pad+1)) {
                        actTemp[kll] = pAct[(idxYY-pad)*yCoeff + (idxXX-pad)*xCoeff + idxZ[4*kl+kll]];
                        xyCount++;
                        mask[kll] = 0xFFFFFFFF;
                     }
                     else {
                        actTemp[kll] = 0;
                        mask[kll] = 0x00000000;
                     }
                  }
                  vecAct = vld1q_s32(actTemp);
                  vecWgt = vld1q_s32(pKrn + index + kl*4);
                  // XNOR
                  vecAct = veorq_s32(vecAct, vecWgt);
                  vecAct = vmvnq_s32(vecAct);
                  // Mask
                  vecAct = vandq_s32(vecAct, mask);
                  // popcount
                  // popcount only works on 8-bit vectors, so needs some casting
                  // Not a problem here because those are binary vectors not values
                  vecAct = vreinterpretq_s32_s8(vcntq_s8(vreinterpretq_s8_s32(vecAct)));
                  // Now we need to do addition
                  // 16x8b reduce to 8x16b
                  vecAct = vreinterpretq_s32_s16(vpaddlq_s8(vreinterpretq_s8_s32(vecAct)));
                  // 8x16b to 4x32b
                  vecAct = vpaddlq_s16(vreinterpretq_s16_s32(vecAct));
                  // 4x32b to a two values
                  vecOut += vpaddlq_s32(vecAct);
               } // K-len
               // Extract the output
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // Adjust the output value
               outTemp = outTemp - (xyCount*pckWdt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *thresh;
               // Shift based on current kernel slice
               outTemp = outTemp << (pckWdt-1-ks);
               // First time writing to a given word, make sure to clear it
               // Write out
               pOut[y*(wdth-kwdt+2*pad+1)*knum/pckWdt + x*knum/pckWdt + k] |= outTemp;
            }
         }
         index += kLen;
         // Each kernel has its own threshold
         thresh++;
      }
   }

   // Adjust signs
   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Grab a batch of bn signs for kernels
      signs = *sign++;
      // Y dim
      for (uint16_t y = 0; y < ((hght-khgt+2*pad)+1); y++) {
         // X dim
         for (uint16_t x = 0; x < ((wdth-kwdt+2*pad)+1); x++) {
            pOut[y*(((wdth-kwdt+2*pad))+1)*knum/pckWdt + x*knum/pckWdt + k] = ~(pOut[y*(((wdth-kwdt+2*pad))+1)*knum/pckWdt + x*knum/pckWdt + k] ^ signs);
         }
      }
   }

   free(idxY);
   free(idxX);
   free(idxZ);
}
#endif

/**
 * @details 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/pooling/batch norm.
 * Kernel as output loop dimension. Batchnorm implemented as a threshold function.
 * 
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix 
 * @param[in] kLen - sparse kernel size 
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] pad  - padding size 
 * @param[in] pool - pooling window size 
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdPl3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   uint8_t  *idxY = malloc((kLen/pckWdt)*sizeof(uint8_t));
   uint8_t  *idxX = malloc((kLen/pckWdt)*sizeof(uint8_t));
   uint8_t  *idxZ = malloc((kLen/pckWdt)*sizeof(uint8_t));
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = kdpt/pckWdt;
   uint16_t index = 0;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   uint16_t  idxYY = 0;
   uint16_t  idxXX = 0;
   // For maxpooling
   int32_t  maxTemp = 0;
   pckDtype  signs = 0;

   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Packed slices
      for (uint16_t ks = 0; ks<pckWdt; ks++) {
         // Loop through all indices and pack them locally
         for (uint8_t kll = 0; kll < kLen/pckWdt; kll++) {
            idxY[kll] = pInd[index + kll]/(kwdt*kCoeff);
            idxX[kll] = (pInd[index + kll]-idxY[kll]*kwdt*kCoeff)/kCoeff;
            idxZ[kll] = pInd[index + kll]-idxY[kll]*kwdt*kCoeff- idxX[kll]*kCoeff;
         }
         // Y dim
         for (uint16_t y = 0; y < ((hght-khgt+2*pad+1)/pool); y++) {
            // X dim
            for (uint16_t x = 0; x < ((wdth-kwdt+2*pad+1)/pool); x++) {
               maxTemp = -(khgt*kwdt*kdpt);
               // Need to do that because we'll be oring into it 
               if (ks == 0) {
                  pOut[y*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt + x*knum/pckWdt + k] = 0;
               }
               for (uint16_t yy = 0; yy < pool; yy++) {
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     outTemp = 0;
                     xyCount = 0;
                     for (uint16_t kl = 0; kl < kLen/pckWdt; kl++) { 
                        idxYY = y*pool+yy+idxY[kl];
                        idxXX = x*pool+xx+idxX[kl];
                        if (!(idxYY < pad || idxXX < pad || idxYY-pad >= hght || idxXX-pad >= wdth)) {
                           xyCount++;
                           // XNOR multiplication
                           xnorTmp = ~( pAct[(idxYY-pad)*yCoeff + (idxXX-pad)*xCoeff + idxZ[kl]] ^ pKrn[(index + kl)]);
                           // popcount
                           xnorTmp = popcount(xnorTmp);
                           // Accumulation
                           outTemp += xnorTmp;
                        }
                     } // K-len
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*pckWdt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = (bnPrec) maxTemp >= *thresh;
               // Shift based on current kernel slice
               maxTemp = maxTemp << (pckWdt-1-ks);
               // First time writing to a given word, make sure to clear it
               // Write out
               pOut[y*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt + x*knum/pckWdt + k] |= maxTemp;
            }
         }
         index += kLen/pckWdt;
         thresh++;
      }
   }

   // Adjust signs
   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Grab a batch of bn signs for kernels
      signs = *sign++;
      // Y dim
      for (uint16_t y = 0; y < ((hght-khgt+2*pad+1)/pool); y++) {
         // X dim
         for (uint16_t x = 0; x < ((wdth-kwdt+2*pad+1)/pool); x++) {
            pOut[y*(((wdth-kwdt+2*pad+1)/pool))*knum/pckWdt + x*knum/pckWdt + k] = ~(pOut[y*(((wdth-kwdt+2*pad+1)/pool))*knum/pckWdt + x*knum/pckWdt + k] ^ signs);
         }
      }
   }
    free(idxY);
    free(idxX);
    free(idxZ);
}


void CnBnPdPl3pxnNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, uint8_t* __restrict pInd, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {

    // Temporary variables
    pckDtype xnorTmp = 0;
    int32_t  outTemp = 0;
    uint8_t* idxY = malloc((kLen / pckWdt) * sizeof(uint8_t));
    uint8_t* idxX = malloc((kLen / pckWdt) * sizeof(uint8_t));
    uint8_t* idxZ = malloc((kLen / pckWdt) * sizeof(uint8_t));
    uint16_t  yCoeff = wdth * dpth / pckWdt;
    uint16_t  xCoeff = dpth / pckWdt;
    uint16_t  kCoeff = kdpt / pckWdt;
    uint16_t index = 0;
    // XY count for padding adjustment
    uint8_t  xyCount = 0;
    uint8_t  idxYY = 0;
    uint8_t  idxXX = 0;
    // For maxpooling
    int32_t  maxTemp = 0;
    pckDtype  signs = 0;

    // Outer loop - kernels
    for (uint16_t k = 0; k < knum / pckWdt; k++) {
        // Packed slices
        for (uint16_t ks = 0; ks < pckWdt; ks++) {
            // Loop through all indices and pack them locally
            for (uint8_t kll = 0; kll < kLen / pckWdt; kll++) {
                idxY[kll] = pInd[index + kll] / (kwdt * kCoeff);
                idxX[kll] = (pInd[index + kll] - idxY[kll] * kwdt * kCoeff) / kCoeff;
                idxZ[kll] = pInd[index + kll] - idxY[kll] * kwdt * kCoeff - idxX[kll] * kCoeff;
            }
            // Y dim
            for (uint16_t y = 0; y < ((hght - khgt + 2 * pad + 1) / pool); y++) {
                // X dim
                for (uint16_t x = 0; x < ((wdth - kwdt + 2 * pad + 1) / pool); x++) {
                    maxTemp = -(khgt * kwdt * kdpt);
                    // Need to do that because we'll be oring into it 
                    if (ks == 0) {
                        pOut[y * ((wdth - kwdt + 2 * pad + 1) / pool) * knum / pckWdt + x * knum / pckWdt + k] = 0;
                    }
                    for (uint16_t yy = 0; yy < pool; yy++) {
                        for (uint16_t xx = 0; xx < pool; xx++) {
                            outTemp = 0;
                            xyCount = 0;
                            for (uint16_t kl = 0; kl < kLen / pckWdt; kl++) {
                                idxYY = y * pool + yy + idxY[kl];
                                idxXX = x * pool + xx + idxX[kl];
                                if (!(idxYY < pad || idxXX < pad || idxYY - pad >= hght || idxXX - pad >= wdth)) {
                                    xyCount++;
                                    // XNOR multiplication
                                    xnorTmp = ~(pAct[(idxYY - pad) * yCoeff + (idxXX - pad) * xCoeff + idxZ[kl]] ^ pKrn[(index + kl)]);
                                    // popcount
                                    xnorTmp = popcount(xnorTmp);
                                    // Accumulation
                                    outTemp += xnorTmp;
                                }
                            } // K-len
                            // Adjust the output value
                            outTemp = outTemp - (xyCount * pckWdt - outTemp);
                            // Maxpool
                            if (outTemp > maxTemp) { maxTemp = outTemp; }
                        }
                    }
                    // Binarize
                    //maxTemp = (bnPrec)maxTemp >= *thresh;
                    // Shift based on current kernel slice
                    //maxTemp = maxTemp << (pckWdt - 1 - ks);
                    // First time writing to a given word, make sure to clear it
                    // Write out
                    pOut[y * (wdth - kwdt + 1) * knum / pckWdt + x * knum / pckWdt + k] = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                }
            }
            index += kLen / pckWdt;
            //thresh++;
        }
    }
    free(idxY);
    free(idxX);
    free(idxZ);
}

#ifdef NEON
/**
 * @details 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/pooling/batch norm/NEON
 * Kernel as output loop dimension. Batchnorm implemented as a threshold function.
 * 
 * @param[in] pAct - pointer to the packed activation vector (detpth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] pInd - pointer to the compressed index matrix 
 * @param[in] kLen - sparse kernel size 
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] pad  - padding size 
 * @param[in] pool - pooling window size 
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdPl3pxnNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, const uint8_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   uint8_t  *idxY = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxX = malloc(kLen*sizeof(uint8_t));
   uint8_t  *idxZ = malloc(kLen*sizeof(uint8_t));
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = kdpt/pckWdt;
   uint16_t index = 0;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   uint8_t  idxYY = 0;
   uint8_t  idxXX = 0;
   // For maxpooling
   int32_t  maxTemp = 0;
   pckDtype  signs = 0;
   int8_t    signCur  = 0;
   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int32x4_t mask;
   int64x2_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[4];

   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Packed slices
      for (uint16_t ks = 0; ks<pckWdt; ks++) {
         // Loop through all indices and pack them locally
         for (uint8_t kll = 0; kll < kLen; kll++) {
            idxY[kll] = pInd[index + kll]/(kwdt*kCoeff);
            idxX[kll] = (pInd[index + kll]-idxY[kll]*kwdt*kCoeff)/kCoeff;
            idxZ[kll] = pInd[index + kll]-idxY[kll]*kwdt*kCoeff- idxX[kll]*kCoeff;
         }
         // Y dim
         for (uint16_t y = 0; y < ((hght-khgt+2*pad)/pool+1); y++) {
            // X dim
            for (uint16_t x = 0; x < ((wdth-kwdt+2*pad)/pool+1); x++) {
               maxTemp = -(khgt*kwdt*kdpt);
               // Need to do that because we'll be oring into it 
               if (ks == 0) {
                  pOut[y*((wdth-kwdt+2*pad)/pool+1)*knum/pckWdt + x*knum/pckWdt + k] = 0;
               }
               for (uint16_t yy = 0; yy < pool; yy++) {
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     xyCount = 0;
                     vecOut[0] = 0;
                     vecOut[1] = 0;
                     for (uint16_t kl = 0; kl < kLen/4; kl++) { 
                        for (uint8_t kll = 0; kll < 4; kll++) {
                           // Check indices
                           idxYY = y*pool+yy+idxY[4*kl+kll];
                           idxXX = x*pool+xx+idxX[4*kl+kll];
                           // Load the values
                           if (!(idxYY < pad || idxXX < pad || idxYY > hght-khgt+2*pad+1 || idxXX > wdth-kwdt+2*pad+1)) {
                              actTemp[kll] = pAct[(idxYY-pad)*yCoeff + (idxXX-pad)*xCoeff + idxZ[4*kl+kll]];
                              xyCount++;
                              mask[kll] = 0xFFFFFFFF;
                           }
                           else {
                              actTemp[kll] = 0;
                              mask[kll] = 0x00000000;
                           }
                        }
                        vecAct = vld1q_s32(actTemp);
                        vecWgt = vld1q_s32(pKrn + index + kl*4);
                        // XNOR
                        vecAct = veorq_s32(vecAct, vecWgt);
                        vecAct = vmvnq_s32(vecAct);
                        // Mask
                        vecAct = vandq_s32(vecAct, mask);
                        // popcount
                        // popcount only works on 8-bit vectors, so needs some casting
                        // Not a problem here because those are binary vectors not values
                        vecAct = vreinterpretq_s32_s8(vcntq_s8(vreinterpretq_s8_s32(vecAct)));
                        // Now we need to do addition
                        // 16x8b reduce to 8x16b
                        vecAct = vreinterpretq_s32_s16(vpaddlq_s8(vreinterpretq_s8_s32(vecAct)));
                        // 8x16b to 4x32b
                        vecAct = vpaddlq_s16(vreinterpretq_s16_s32(vecAct));
                        // 4x32b to a two values
                        vecOut += vpaddlq_s32(vecAct);
                     } // K-len
                     // Extract the output
                     outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*pckWdt - outTemp);
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Batch normalize/ binarize
               maxTemp = (bnPrec) maxTemp >= *thresh;
               // Shift based on current kernel slice
               maxTemp = maxTemp << (pckWdt-1-ks);
               // First time writing to a given word, make sure to clear it
               // Write out
               pOut[y*((wdth-kwdt+2*pad)/pool+1)*knum/pckWdt + x*knum/pckWdt + k] |= maxTemp;
            }
         }
         index += kLen;
         // Each kernel has its own threshold
         thresh++;
      }
   }

   // Adjust signs
   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Grab a batch of bn signs for kernels
      signs = *sign++;
      // Y dim
      for (uint16_t y = 0; y < ((hght-khgt+2*pad)/pool+1); y++) {
         // X dim
         for (uint16_t x = 0; x < ((wdth-kwdt+2*pad)/pool+1); x++) {
            pOut[y*(((wdth-kwdt+2*pad)/pool)+1)*knum/pckWdt + x*knum/pckWdt + k] = ~(pOut[y*(((wdth-kwdt+2*pad)/pool)+1)*knum/pckWdt + x*knum/pckWdt + k] ^ signs);
         }
      }
   }
   
   free(idxY);
   free(idxX);
   free(idxZ);
//end:;
}

#endif


