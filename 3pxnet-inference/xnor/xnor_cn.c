/*
* MIT License
* 
* Copyright (c) 2019 UCLA NanoCAD Laboratory 
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without __restriction, including without limitation the rights
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
 * \file      xnor_dense_cn.c
 * \brief     Dense binarized (XNOR) convolutional layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#include "xnor_cn.h"

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - general wrapper.
 * Selects the appropriate implementation (batch normalization, NEON support)
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] pool - pooling size
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 * @return 0 - Success, 1 - Failure
 */
uint8_t CnXnorWrap(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint16_t pad, const uint16_t pool, bnDtype * __restrict thresh, pckDtype * sign) {

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
         // No padding
         if (pad == 0) {
            // No pooling
            if (pool == 1) {
               CnBnXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, thresh, sign);
            }
            // Pooling
            else {
               CnBnPlXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pool, thresh, sign);
            }
         }
         // Padding
         else {
            // No pooling
            if (pool == 1) {
               CnBnPdXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, thresh, sign);
            }
            // Pooling
            else {
               CnBnPdPlXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool, thresh, sign);
            }

         }
         return 0;
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
               CnXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut);
               //CnXnorKOut(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut);
            }
            // Pooling
            else {
               CnPlXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pool);
            }
         }
         // Padding
         else {
            // No pooling
            if (pool == 1) {
               CnPdXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad);
            }
            // Pooling
            else {
               CnPdPlXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool);
            }

         }
         return 0;
      }
   }

}



uint8_t CnXnorNoBinWrap(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint16_t pad, const uint16_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {

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
            // No padding
            if (pad == 0) {
                // No pooling
                if (pool == 1) {
                    CnBnXnorNoBin(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, mean, var, gamma, beta);
                }
                // Pooling
                else {
                    CnBnPlXnorNoBin(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pool, mean,var,gamma,beta);
                }
            }
            // Padding
            else {
                // No pooling
                if (pool == 1) {
                    CnBnPdXnorNoBin(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, mean, var,gamma,beta);
                }
                // Pooling
                else {
                    CnBnPdPlXnorNoBin(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool, mean, var,gamma,beta);
                }

            }
            return 0;
        }
    }

}



/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop
 * Outer loop: xy, Pad: no, Pool: no BatchNorm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 */
void CnXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  cntCoeff = khgt*kwdt*kdpt/2;
   
   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1); y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
         // Outer loop - kernels
         pWgt = pKrn;   
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               pIn = pAct + y*yCoeff + x*xCoeff;
               outTemp = 0;
               // K-Y dim
               for (uint16_t ky = 0; ky < khgt; ky++) {
                  // K-X dim
                  for (uint16_t kx = 0; kx < kwdt*dpth/pckWdt; kx++) {
                     // XNOR multiplication
                     xnorTmp = ~ ( *pIn++ ^ *pWgt++ ); 
                     outTemp += popcount(xnorTmp);
                  } // K-X dim
                  // Move the activation pointer one row down
                  pIn += (wdth-kwdt)*dpth/pckWdt;
               } // K-Y dim
               // We've only counted ones, but we want a difference between +1s and -1s 
               // so we need to adjust the result
               // Below is shorter for
               // outTemp = outTemp - (2*cntCoeff - outTemp);
               // outTemp = outTemp >= 0;
               outTemp = outTemp >= cntCoeff;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
            }
            *pRes++ = pckTemp;
         }
      }
   }
}

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, padding
 * Outer loop: xy, Pad: yes, Pool: no BatchNorm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 */
void CnPdXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   uint16_t  cntCoeff = khgt*kwdt*kdpt/2;
   // Starting indices for padding
   uint16_t  xStart, yStart = 0;
   // Ending indices for padding
   uint16_t  xEnd, yEnd = 0;

   // Divide the input into 5 regions - top, bottom, left, right, middle 
   // Middle has no padding

   // Middle - no padding
   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1); y++) {
      // X dim
      // Set the output pointer
      // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
      // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
      // Offset to this row pad*knum/pckWdt
      pRes = pOut + (pad+y)*(hght-khgt+2*pad+1)*knum/pckWdt + pad*knum/pckWdt;
      for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
         // Outer loop - kernels
         pWgt = pKrn;   
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               pIn = pAct + y*yCoeff + x*xCoeff;
               outTemp = 0;
               // K-Y dim
               for (uint16_t ky = 0; ky < khgt; ky++) {
                  // K-X dim
                  for (uint16_t kx = 0; kx < kwdt*dpth/pckWdt; kx++) {
                     // XNOR multiplication
                     xnorTmp = ~ ( *pIn++ ^ *pWgt++ ); 
                     outTemp += popcount(xnorTmp);
                  } // K-X dim
                  // Move the activation pointer one row down
                  pIn += (wdth-kwdt)*dpth/pckWdt;
               } // K-Y dim
               //goto end;
               // We've only counted ones, but we want a difference between +1s and -1s 
               // so we need to adjust the result
               // Below is shorter for
               // outTemp = outTemp - (2*cntCoeff - outTemp);
               // outTemp = outTemp >= 0;
               outTemp = outTemp >= cntCoeff;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
            }
            *pRes++ = pckTemp;
         }
      }
   }

   // Top
   pRes = pOut;
   // Y dim
   for (uint16_t y = 0; y < pad; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               outTemp = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                           // popcount
                           outTemp += popcount(xnorTmp);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Binarize
               outTemp = outTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += yStart*kwdt*kdpt/pckWdt;
            }
            *pRes++ = pckTemp;
         }
      }
   }
   
   // Bottom 
   // Move the ouput pointer
   pRes = pOut + (hght-khgt+pad+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = hght-khgt+pad+1; y < hght-khgt+2*pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               outTemp = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                           // popcount
                           // Accumulation
                           outTemp += popcount(xnorTmp);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Binarize
               outTemp = outTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            *pRes++ = pckTemp;
         }
      }
   }
  
   // Left 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < pad; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               outTemp = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                           // popcount
                           // Accumulation
                           outTemp += popcount(xnorTmp);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Binarize
               outTemp = outTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   }

   // Right 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum/pckWdt + (wdth-kwdt+pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = wdth-kwdt+pad+1; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               outTemp = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                           // popcount
                           // Accumulation
                           outTemp += popcount(xnorTmp);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Binarize
               outTemp = outTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt + (wdth-kwdt+pad+1)*knum/pckWdt;
   }
}


/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, pooling
 * Outer loop: xy, Pad: no, Pool: yes, BatchNorm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pool - pooling size
 */
void CnPlXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pool) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   // For maxpooling
   int32_t  maxTemp = 0;
   //int32_t  *outTemp = malloc(pool*pool*sizeof(int32_t));
   pckDtype pckTemp = 0;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;

   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1)/pool; y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+1)/pool; x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(khgt*kwdt*kdpt);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     outTemp = 0;
                     pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt;
                     pIn = pAct + (y*pool+yy)*yCoeff + (x*pool+xx)*xCoeff;
                     // K-Y dim
                     for (uint16_t ky = 0; ky < khgt; ky++) {
                        // K-X dim
                        for (uint16_t kx = 0; kx < kwdt; kx++) {
                           // Z dim
                           for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                              // XNOR multiplication
                              xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                              // popcount
                              // Accumulation
                              outTemp += popcount(xnorTmp);
                           } // Z dim
                        } // K-X dim
                        pIn += (wdth-kwdt)*dpth/pckWdt;
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (khgt*kwdt*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  } // X-MP
               } // Y-MP
               // Binarize
               maxTemp = maxTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            *pRes++ = pckTemp;
         }
      }
   }
}


/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, padding/pooling
 * Outer loop: xy, Pad: yes, Pool: yes, BatchNorm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] pool - pooling size
 */
void CnPdPlXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   uint16_t  cntCoeff = khgt*kwdt*kdpt;
   // Starting indices for padding
   uint16_t  xStart, yStart = 0;
   // Ending indices for padding
   uint16_t  xEnd, yEnd = 0;
   // For maxpooling
   int32_t  maxTemp = 0;

   // Divide the input into 5 regions - top, bottom, left, right, middle 
   // Middle has no padding

   // Middle - no padding
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= (hght-khgt+2*pad+1)/pool - 2*((pad+pool-1)/pool); y++) {
      // X dim
      // Set the output pointer
      // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
      // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
      // Offset to this row pad*knum/pckWdt
      pRes = pOut + (y)*((hght-khgt+2*pad+1)/pool)*knum/pckWdt + ((pad+pool-1)/pool)*knum/pckWdt;
      for (uint16_t x = ((pad+pool-1)/pool); x <= (wdth-kwdt+2*pad+1)/pool - 2*((pad+pool-1)/pool); x++) {
         // Outer loop - kernels
         pWgt = pKrn;   
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(khgt*kwdt*kdpt);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     pIn = pAct + (y*pool+yy - pad)*yCoeff + (x*pool+xx- pad)*xCoeff;
                     pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt;
                     outTemp = 0;
                     // K-Y dim
                     for (uint16_t ky = 0; ky < khgt; ky++) {
                        // K-X dim
                        for (uint16_t kx = 0; kx < kwdt*dpth/pckWdt; kx++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++ ); 
                           outTemp += popcount(xnorTmp);
                        } // K-X dim
                        // Move the activation pointer one row down
                        pIn += (wdth-kwdt)*dpth/pckWdt;
                     } // K-Y dim
                     outTemp = 2*outTemp - cntCoeff;
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                     // Shift based on current kernel slice
                  } // X-MP
               } // Y-MP
               // Binarize
               maxTemp = maxTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            *pRes++ = pckTemp;
         }
      }
   }

   //// Top
   pRes = pOut;
   // Y dim
   // We need to make sure there's enough lines to do pooling
   //for (uint16_t y = 0; y < pad; y++) {
   for (uint16_t y = 0; y < (pad+pool-1)/pool; y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+2*pad+1)/pool; x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(khgt*kwdt*kdpt);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     outTemp = 0;
                     // Move the wieight pointer to the fisrt useful (non-padded) weight block
                     pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
                     xyCount = 0;
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + ((y*pool+yy)+ky-pad)*yCoeff + ((x*pool+xx)+xStart-pad)*xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                              xyCount++;
                              // Z dim
                              for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                                 // XNOR multiplication
                                 xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                                 // popcount
                                 outTemp += popcount(xnorTmp);
                              } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = maxTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
            }
            *pRes++ = pckTemp;
         }
      }
   }
   
   // Bottom 
   // Move the ouput pointer
   pRes = pOut + ((hght-khgt+2*pad)/pool + 1 - ((pad+pool-1)/pool))*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt;
   // Y dim
   for (uint16_t y = (hght-khgt+2*pad)/pool + 1 - ((pad+pool-1)/pool); y < (hght-khgt+2*pad)/pool + 1; y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+2*pad)/pool +1; x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(khgt*kwdt*kdpt);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     // Move the wieight pointer to the fisrt useful (non-padded) weight block
                     pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
                     outTemp = 0;
                     xyCount = 0;
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + ((y*pool+yy)+ky-pad)*yCoeff + ((x*pool+xx)+xStart-pad)*xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                              xyCount++;
                              // Z dim
                              for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                                 // XNOR multiplication
                                 xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                                 // popcount
                                 // Accumulation
                                 outTemp += popcount(xnorTmp);
                              } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = maxTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            *pRes++ = pckTemp;
         }
      }
   }
  
   //// Left 
   pRes = pOut + ((pad+pool-1)/pool)*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt;
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= (hght-khgt+2*pad+1)/pool - 2*((pad+pool-1)/pool); y++) {
      // X dim
      for (uint16_t x = 0; x < ((pad+pool-1)/pool); x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(khgt*kwdt*kdpt);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     outTemp = 0;
                     xyCount = 0;
                     // Move the wieight pointer to the fisrt useful (non-padded) weight block
                     pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + ((y*pool+yy)+ky-pad)*yCoeff + ((x*pool+xx)+xStart-pad)*xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                              xyCount++;
                              // Z dim
                              for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                                 // XNOR multiplication
                                 xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                                 // popcount
                                 // Accumulation
                                 outTemp += popcount(xnorTmp);
                              } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = maxTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt;
   }

   // Right 
   pRes = pOut + ((pad+pool-1)/pool)*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt + (((wdth-kwdt+2*pad+1)/pool) - ((pad+pool-1)/pool))*knum/pckWdt;
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= (hght-khgt+2*pad+1)/pool - 2*((pad+pool-1)/pool); y++) {
      // X dim
      for (uint16_t x = (wdth-kwdt+2*pad)/pool + 1 - ((pad+pool-1)/pool); x < (wdth-kwdt+2*pad)/pool + 1; x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(khgt*kwdt*kdpt);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     // Move the wieight pointer to the fisrt useful (non-padded) weight block
                     pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
                     outTemp = 0;
                     xyCount = 0;
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + ((y*pool+yy)+ky-pad)*yCoeff + ((x*pool+xx)+xStart-pad)*xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                           xyCount++;
                           // Z dim
                           for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                              // XNOR multiplication
                              xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                              // popcount
                              // Accumulation
                              outTemp += popcount(xnorTmp);
                           } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = maxTemp >= 0;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt + (((wdth-kwdt+2*pad+1)/pool) - ((pad+pool-1)/pool))*knum/pckWdt;
   }
}

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - Kernel outer loop, batch norm
 * Outer loop: kernel, Pad: no, Pool: no BatchNorm: yes, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnXnorKOut(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = khgt*kwdt*kdpt/pckWdt;
   uint16_t  kyCoeff = kwdt*kdpt/pckWdt;
   uint16_t  kxCoeff = kdpt/pckWdt;
   pckDtype  signs = 0;
   int8_t    signCur  = 0;

   // Outer loop - kernels
   for (uint16_t k = 0; k<knum/pckWdt; k++) {
      // Grab a batch of bn signs for kernels
      signs = *sign++;
      // Packed slices
      for (uint16_t ks = 0; ks<pckWdt; ks++) {
         // Unpack current sign
         signCur = (signs >> ks) & 1;
         if (signCur == 0) {signCur = -1;}
         // Y dim
         for (uint16_t y = 0; y < (hght-khgt+1); y++) {
            // X dim
            for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
               outTemp = 0;
               // K-Y dim
               for (uint16_t ky = 0; ky < khgt; ky++) {
                  // K-X dim
                  for (uint16_t kx = 0; kx < kwdt; kx++) {
                     // Z dim
                     for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                        // XNOR multiplication
                        // (y+ky)*wdth*dpth/pckWdt - starting y position
                        // (x+kx)*dpth/pckWdt - starting x position
                        xnorTmp = ~ ( pAct[(y+ky)*yCoeff + (x+kx)*xCoeff + z] ^ pKrn[(k*pckWdt+ks)*kCoeff +  ky*kyCoeff + kx*kxCoeff + z]);
                        // popcount
                        xnorTmp = popcount(xnorTmp);
                        // Accumulation
                        outTemp += xnorTmp;
                     } // Z dim
                  } // K-X dim
               } // K-Y dim
               // Adjust the output value
               outTemp = outTemp - (khgt*kwdt*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *thresh;
               outTemp *= signCur;
               outTemp = outTemp >= 0;
               // Shift based on current kernel slice
               outTemp = outTemp << (pckWdt-1-ks);
               // First time writing to a given word, make sure to clear it
               // Need to do that because we'll be oring into it 
               if (ks == 0) {
                  pOut[y*(wdth-kwdt+1)*knum/pckWdt + x*knum/pckWdt + k] = 0;
               }
               // Write out
               pOut[y*(wdth-kwdt+1)*knum/pckWdt + x*knum/pckWdt + k] |= outTemp;
            }
         }
         // Each kernel has its own threshold
         thresh++;
      }
   }

}

void CnBnXnorKOutNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, 
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, 
    const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, 
    bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {
    // Temporary variables
    pckDtype xnorTmp = 0;
    int32_t  outTemp = 0;
    uint16_t  yCoeff = wdth * dpth / pckWdt;
    uint16_t  xCoeff = dpth / pckWdt;
    uint16_t  kCoeff = khgt * kwdt * kdpt / pckWdt;
    uint16_t  kyCoeff = kwdt * kdpt / pckWdt;
    uint16_t  kxCoeff = kdpt / pckWdt;

    // Outer loop - kernels
    for (uint16_t k = 0; k < knum / pckWdt; k++) {
        // Grab a batch of bn signs for kernels
        // Packed slices
        for (uint16_t ks = 0; ks < pckWdt; ks++) {
            // Y dim
            for (uint16_t y = 0; y < (hght - khgt + 1); y++) {
                // X dim
                for (uint16_t x = 0; x < (wdth - kwdt + 1); x++) {
                    outTemp = 0;
                    // K-Y dim
                    for (uint16_t ky = 0; ky < khgt; ky++) {
                        // K-X dim
                        for (uint16_t kx = 0; kx < kwdt; kx++) {
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                // XNOR multiplication
                                // (y+ky)*wdth*dpth/pckWdt - starting y position
                                // (x+kx)*dpth/pckWdt - starting x position
                                xnorTmp = ~(pAct[(y + ky) * yCoeff + (x + kx) * xCoeff + z] ^ pKrn[(k * pckWdt + ks) * kCoeff + ky * kyCoeff + kx * kxCoeff + z]);
                                // popcount
                                xnorTmp = popcount(xnorTmp);
                                // Accumulation
                                outTemp += xnorTmp;
                            } // Z dim
                        } // K-X dim
                    } // K-Y dim
                    // Adjust the output value
                    outTemp = outTemp - (khgt * kwdt * kdpt - outTemp);
                    // Batch normalize
                    *pOut++ = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                }
            }
            // Each kernel has its own threshold
        }
    }
}
/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm
 * Outer loop: XY, Pad: no, Pool: no BatchNorm: yes, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   pckDtype  *signs = sign;
   bnDtype   *threshLoc = thresh;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  cntCoeff = khgt*kwdt*kdpt/2;
   
   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1); y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         pWgt = pKrn;   
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               pIn = pAct + y*yCoeff + x*xCoeff;
               outTemp = 0;
               // K-Y dim
               for (uint16_t ky = 0; ky < khgt; ky++) {
                  // K-X dim
                  for (uint16_t kx = 0; kx < kwdt*dpth/pckWdt; kx++) {
                     // XNOR multiplication
                     xnorTmp = ~ ( *pIn++ ^ *pWgt++ ); 
                     outTemp += popcount(xnorTmp);
                  } // K-X dim
                  // Move the activation pointer one row down
                  pIn += (wdth-kwdt)*dpth/pckWdt;
               } // K-Y dim
               // We've only counted ones, but we want a difference between +1s and -1s 
               // so we need to adjust the result
               // Below is shorter for
               // outTemp = outTemp - (2*cntCoeff - outTemp);
               // outTemp = outTemp >= 0;
               outTemp = outTemp - (2*cntCoeff-outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
}

void CnBnXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth,
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, 
    const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut,
    bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {

    // Temporary variables
    pckDtype xnorTmp = 0;
    int32_t  outTemp = 0;
    pckDtype pckTemp = 0;
    // Moving kernel pointer
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    uint16_t  yCoeff = wdth * dpth / pckWdt;
    uint16_t  xCoeff = dpth / pckWdt;
    uint16_t  cntCoeff = khgt * kwdt * kdpt / 2;

    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1); y++) {
        // X dim
        for (uint16_t x = 0; x < (wdth - kwdt + 1); x++) {
            // Outer loop - kernels
            pWgt = pKrn;
            for (uint16_t k = 0; k < knum; k++) {
                // Packed slices
                pckTemp = 0;
                pIn = pAct + y * yCoeff + x * xCoeff;
                outTemp = 0;
                // K-Y dim
                for (uint16_t ky = 0; ky < khgt; ky++) {
                    // K-X dim
                    for (uint16_t kx = 0; kx < kwdt * dpth / pckWdt; kx++) {
                        // XNOR multiplication
                        xnorTmp = ~(*pIn++ ^ *pWgt++);
                        outTemp += popcount(xnorTmp);
                    } // K-X dim
                    // Move the activation pointer one row down
                    pIn += (wdth - kwdt) * dpth / pckWdt;
                } // K-Y dim
                // We've only counted ones, but we want a difference between +1s and -1s 
                // so we need to adjust the result
                // Below is shorter for
                // outTemp = outTemp - (2*cntCoeff - outTemp);
                // outTemp = outTemp >= 0;
                outTemp = outTemp - (2 * cntCoeff - outTemp);
                // Batch normalize/ binarize
                float temp = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                pOut[k] = temp;
            }
        }
    }
}

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding
 * Outer loop: XY, Pad: yes, Pool: no BatchNorm: yes, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdXnor(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype* __restrict pOut, const uint8_t pad, bnDtype* __restrict thresh, pckDtype* sign){

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   pckDtype  *signs = sign;
   bnDtype   *threshLoc = thresh;
   uint16_t  cntCoeff = khgt*kwdt*kdpt/2;
   // Starting indices for padding
   uint16_t  xStart, yStart = 0;
   // Ending indices for padding
   uint16_t  xEnd, yEnd = 0;

   // Divide the input into 5 regions - top, bottom, left, right, middle 
   // Middle has no padding

   // Middle - no padding
   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1); y++) {
      // X dim
      // Set the output pointer
      // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
      // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
      // Offset to this row pad*knum/pckWdt
      pRes = pOut + (pad+y)*(hght-khgt+2*pad+1)*knum/pckWdt + pad*knum/pckWdt;
      for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
         // Outer loop - kernels
         pWgt = pKrn;   
         threshLoc = thresh;
         signs = sign;
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               pIn = pAct + y*yCoeff + x*xCoeff;
               outTemp = 0;
               // K-Y dim
               for (uint16_t ky = 0; ky < khgt; ky++) {
                  // K-X dim
                  for (uint16_t kx = 0; kx < kwdt*dpth/pckWdt; kx++) {
                     // XNOR multiplication
                     xnorTmp = ~ ( *pIn++ ^ *pWgt++ ); 
                     outTemp += popcount(xnorTmp);
                  } // K-X dim
                  // Move the activation pointer one row down
                  pIn += (wdth-kwdt)*dpth/pckWdt;
               } // K-Y dim
               // We've only counted ones, but we want a difference between +1s and -1s 
               // so we need to adjust the result
               // Below is shorter for
               // outTemp = outTemp - (2*cntCoeff - outTemp);
               // outTemp = outTemp >= 0;
               outTemp = outTemp - (2*cntCoeff - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }

   // Top
   pRes = pOut;
   // Y dim
   for (uint16_t y = 0; y < pad; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               outTemp = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                           // popcount
                           outTemp += popcount(xnorTmp);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += yStart*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
   
   // Bottom 
   // Move the ouput pointer
   pRes = pOut + (hght-khgt+pad+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = hght-khgt+pad+1; y < hght-khgt+2*pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               outTemp = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                           // popcount
                           // Accumulation
                           outTemp += popcount(xnorTmp);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
  
   // Left 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < pad; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               outTemp = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                           // popcount
                           // Accumulation
                           outTemp += popcount(xnorTmp);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   }

   // Right 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum/pckWdt + (wdth-kwdt+pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = wdth-kwdt+pad+1; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               outTemp = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                           // popcount
                           // Accumulation
                           outTemp += popcount(xnorTmp);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt + (wdth-kwdt+pad+1)*knum/pckWdt;
   }
}

void CnBnPdXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad,bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {
    // Temporary variables
    pckDtype xnorTmp = 0;
    int32_t  outTemp = 0;
    pckDtype pckTemp = 0;
    uint16_t  yCoeff = wdth * dpth / pckWdt;
    uint16_t  xCoeff = dpth / pckWdt;
    // XY count for padding adjustment
    uint8_t  xyCount = 0;
    // Moving kernel pointer
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    bnDtype* pRes = pOut;
    uint16_t  cntCoeff = khgt * kwdt * kdpt / 2;
    // Starting indices for padding
    uint16_t  xStart, yStart = 0;
    // Ending indices for padding
    uint16_t  xEnd, yEnd = 0;

    // Divide the input into 5 regions - top, bottom, left, right, middle 
    // Middle has no padding

    // Middle - no padding
    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1); y++) {
        // X dim
        // Set the output pointer
        // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
        // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
        // Offset to this row pad*knum/pckWdt
        pRes = pOut + (pad + y) * (hght - khgt + 2 * pad + 1) * knum / pckWdt + pad * knum / pckWdt;
        for (uint16_t x = 0; x < (wdth - kwdt + 1); x++) {
            // Outer loop - kernels
            pWgt = pKrn;
            //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                pckTemp = 0;
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    pIn = pAct + y * yCoeff + x * xCoeff;
                    outTemp = 0;
                    // K-Y dim
                    for (uint16_t ky = 0; ky < khgt; ky++) {
                        // K-X dim
                        for (uint16_t kx = 0; kx < kwdt * dpth / pckWdt; kx++) {
                            // XNOR multiplication
                            xnorTmp = ~(*pIn++ ^ *pWgt++);
                            outTemp += popcount(xnorTmp);
                        } // K-X dim
                        // Move the activation pointer one row down
                        pIn += (wdth - kwdt) * dpth / pckWdt;
                    } // K-Y dim
                    // We've only counted ones, but we want a difference between +1s and -1s 
                    // so we need to adjust the result
                    // Below is shorter for
                    // outTemp = outTemp - (2*cntCoeff - outTemp);
                    // outTemp = outTemp >= 0;
                    outTemp = outTemp - (2 * cntCoeff - outTemp);
                    // Batch normalize/ binarize
                    *pRes++ = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                    // Shift based on current kernel slice
                }
            }
        }
    }

    // Top
    pRes = pOut;
    // Y dim
    for (uint16_t y = 0; y < pad; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = 0; x < wdth - kwdt + 2 * pad + 1; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                pckTemp = 0;
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    outTemp = 0;
                    xyCount = 0;
                    // K-Y dim
                    for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                            xyCount++;
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                // XNOR multiplication
                                xnorTmp = ~(*pIn++ ^ *pWgt++);
                                // popcount
                                outTemp += popcount(xnorTmp);
                            } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                    } // K-Y dim
                    // Adjust the output value
                    outTemp = outTemp - (xyCount * kdpt - outTemp);
                    // Batch normalize/ binarize
                    *pRes++ = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                    // Shift the weight pointer to the next kernel
                    pWgt += yStart * kwdt * kdpt / pckWdt;
                }
            }
        }
    }

    // Bottom 
    // Move the ouput pointer
    pRes = pOut + (hght - khgt + pad + 1) * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt;
    // Y dim
    for (uint16_t y = hght - khgt + pad + 1; y < hght - khgt + 2 * pad + 1; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = 0; x < wdth - kwdt + 2 * pad + 1; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                pckTemp = 0;
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    outTemp = 0;
                    xyCount = 0;
                    // K-Y dim
                    for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                            xyCount++;
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                // XNOR multiplication
                                xnorTmp = ~(*pIn++ ^ *pWgt++);
                                // popcount
                                // Accumulation
                                outTemp += popcount(xnorTmp);
                            } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                    } // K-Y dim
                    // Adjust the output value
                    outTemp = outTemp - (xyCount * kdpt - outTemp);
                    // Batch normalize/ binarize
                    *pRes++ = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                    // Shift the weight pointer to the next kernel
                    pWgt += (khgt - yEnd + yStart) * kwdt * kdpt / pckWdt;
                }
            }
        }
    }

    // Left 
    pRes = pOut + pad * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt;
    // Y dim
    for (uint16_t y = pad; y < hght - khgt + pad + 1; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = 0; x < pad; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                pckTemp = 0;
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    outTemp = 0;
                    xyCount = 0;
                    // K-Y dim
                    for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                            xyCount++;
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                // XNOR multiplication
                                xnorTmp = ~(*pIn++ ^ *pWgt++);
                                // popcount
                                // Accumulation
                                outTemp += popcount(xnorTmp);
                            } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                    } // K-Y dim
                    // Adjust the output value
                    outTemp = outTemp - (xyCount * kdpt - outTemp);
                    // Batch normalize/ binarize
                    *pRes++ = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                    // Shift the weight pointer to the next kernel
                    pWgt += (khgt - yEnd + yStart) * kwdt * kdpt / pckWdt;
                }
            }
        }
        pRes = pOut + (y + 1) * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt;
    }

    // Right 
    pRes = pOut + pad * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt + (wdth - kwdt + pad + 1) * knum / pckWdt;
    // Y dim
    for (uint16_t y = pad; y < hght - khgt + pad + 1; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = wdth - kwdt + pad + 1; x < wdth - kwdt + 2 * pad + 1; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                pckTemp = 0;
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    outTemp = 0;
                    xyCount = 0;
                    // K-Y dim
                    for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                            xyCount++;
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                // XNOR multiplication
                                xnorTmp = ~(*pIn++ ^ *pWgt++);
                                // popcount
                                // Accumulation
                                outTemp += popcount(xnorTmp);
                            } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                    } // K-Y dim
                    // Adjust the output value
                    outTemp = outTemp - (xyCount * kdpt - outTemp);
                    // Batch normalize/ binarize
                    *pRes++ = (float)*gamma++ * (((bnPrec)outTemp - *mean++) / (*var++)) + *beta++;
                    // Shift the weight pointer to the next kernel
                    pWgt += (khgt - yEnd + yStart) * kwdt * kdpt / pckWdt;
                }
            }
        }
        pRes = pOut + (y + 1) * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt + (wdth - kwdt + pad + 1) * knum / pckWdt;
    }
}


#ifdef NEON
/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding, NEON
 * Outer loop: XY, Pad: yes, Pool: no BatchNorm: yes, SIMD: NEON (128) 
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdXnorNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = khgt*kwdt*kdpt/pckWdt;
   uint16_t  kyCoeff = kwdt*kdpt/pckWdt;
   uint16_t  kxCoeff = kdpt/pckWdt;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   // Output X/Y dimensions
   uint16_t outYDim = hght-khgt+2*pad+1;
   uint16_t outXDim = wdth-kwdt+2*pad+1;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   pckDtype  *signs = sign;
   bnDtype   *threshLoc = thresh;
   uint16_t  cntCoeff = khgt*kwdt*kdpt/2;
   // Starting indices for padding
   uint16_t  xStart, yStart = 0;
   // Ending indices for padding
   uint16_t  xEnd, yEnd = 0;
   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int64x2_t vecOut ;

   // Divide the input into 5 regions - top, bottom, left, right, middle 
   // Middle has no padding

   // Middle - no padding
   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1); y++) {
      // X dim
      // Set the output pointer
      // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
      // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
      // Offset to this row pad*knum/pckWdt
      pRes = pOut + (pad+y)*(hght-khgt+2*pad+1)*knum/pckWdt + pad*knum/pckWdt;
      for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
         // Outer loop - kernels
         pWgt = pKrn;   
         threshLoc = thresh;
         signs = sign;
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               pIn = pAct + y*yCoeff + x*xCoeff;
               vecOut[0] = 0;
               vecOut[1] = 0;
               // K-Y dim
               for (uint16_t ky = 0; ky < khgt; ky++) {
                  // K-X dim
                  for (uint16_t kx = 0; kx < kwdt*(dpth/128)/pckWdt; kx++) {
                     // Load values
                     vecAct = vld1q_s32(*pIn);
                     vecWgt = vld1q_s32(*pWgt);
                     pIn += 4;
                     pWgt += 4;
                     // XNOR
                     vecAct = veorq_s32(vecAct, vecWgt);
                     vecAct = vmvnq_s32(vecAct);
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
                  } // K-X dim
                  // Move the activation pointer one row down
                  pIn += (wdth-kwdt)*dpth/pckWdt;
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // We've only counted ones, but we want a difference between +1s and -1s 
               // so we need to adjust the result
               // Below is shorter for
               // outTemp = outTemp - (2*cntCoeff - outTemp);
               // outTemp = outTemp >= 0;
               outTemp = outTemp - cntCoeff;
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }

   // Top
   pRes = pOut;
   // Y dim
   for (uint16_t y = 0; y < pad; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               vecOut[0] = 0;
               vecOut[1] = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < (dpth/128)/pckWdt; z++) {
                           // Load values
                           vecAct = vld1q_s32(*pIn);
                           vecWgt = vld1q_s32(*pWgt);
                           pIn += 4;
                           pWgt += 4;
                           // XNOR
                           vecAct = veorq_s32(vecAct, vecWgt);
                           vecAct = vmvnq_s32(vecAct);
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
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += yStart*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
   
   // Bottom 
   // Move the ouput pointer
   pRes = pOut + (hght-khgt+pad+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = hght-khgt+pad+1; y < hght-khgt+2*pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               vecOut[0] = 0;
               vecOut[1] = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < (dpth/128)/pckWdt; z++) {
                           // Load values
                           vecAct = vld1q_s32(*pIn);
                           vecWgt = vld1q_s32(*pWgt);
                           pIn += 4;
                           pWgt += 4;
                           // XNOR
                           vecAct = veorq_s32(vecAct, vecWgt);
                           vecAct = vmvnq_s32(vecAct);
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
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
  
   // Left 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < pad; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               vecOut[0] = 0;
               vecOut[1] = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < (dpth/128)/pckWdt; z++) {
                           // Load values
                           vecAct = vld1q_s32(*pIn);
                           vecWgt = vld1q_s32(*pWgt);
                           pIn += 4;
                           pWgt += 4;
                           // XNOR
                           vecAct = veorq_s32(vecAct, vecWgt);
                           vecAct = vmvnq_s32(vecAct);
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
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   }

   // Right 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum/pckWdt + (wdth-kwdt+pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = wdth-kwdt+pad+1; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               vecOut[0] = 0;
               vecOut[1] = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < (dpth/128)/pckWdt; z++) {
                           // Load values
                           vecAct = vld1q_s32(*pIn);
                           vecWgt = vld1q_s32(*pWgt);
                           pIn += 4;
                           pWgt += 4;
                           // XNOR
                           vecAct = veorq_s32(vecAct, vecWgt);
                           vecAct = vmvnq_s32(vecAct);
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
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt + (wdth-kwdt+pad+1)*knum/pckWdt;
   }
}
#endif

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, pooling
 * Outer loop: XY, Pad: no, Pool: yes BatchNorm: yes, SIMD: no  
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pool - pooling size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPlXnor(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype* __restrict pOut, const uint8_t pool, bnDtype* __restrict thresh, pckDtype* sign){

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   // For maxpooling
   int32_t  maxTemp = 0;
   //int32_t  *outTemp = malloc(pool*pool*sizeof(int32_t));
   pckDtype pckTemp = 0;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   pckDtype  *signs = sign;
   bnDtype   *threshLoc = thresh;

   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1)/pool; y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+1)/pool; x++) {
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(khgt*kwdt*kdpt);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     outTemp = 0;
                     pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt;
                     pIn = pAct + (y*pool+yy)*yCoeff + (x*pool+xx)*xCoeff;
                     // K-Y dim
                     for (uint16_t ky = 0; ky < khgt; ky++) {
                        // K-X dim
                        for (uint16_t kx = 0; kx < kwdt; kx++) {
                           // Z dim
                           for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                              // XNOR multiplication
                              xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                              // popcount
                              // Accumulation
                              outTemp += popcount(xnorTmp);
                           } // Z dim
                        } // K-X dim
                        pIn += (wdth-kwdt)*dpth/pckWdt;
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (khgt*kwdt*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  } // X-MP
               } // Y-MP
               // Batch normalize/ binarize
               //goto end;
               maxTemp = (bnPrec) maxTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
}

void CnBnPlXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, 
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, 
    const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, 
    const uint8_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {
    // Temporary variables
    pckDtype xnorTmp = 0;
    int32_t  outTemp = 0;
    // For maxpooling
    int32_t  maxTemp = 0;
    //int32_t  *outTemp = malloc(pool*pool*sizeof(int32_t));
    pckDtype pckTemp = 0;
    uint16_t  yCoeff = wdth * dpth / pckWdt;
    uint16_t  xCoeff = dpth / pckWdt;
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    bnDtype* pRes = pOut;
    
    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1) / pool; y++) {
        // X dim
        for (uint16_t x = 0; x < (wdth - kwdt + 1) / pool; x++) {
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                pckTemp = 0;
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    // Mpool patches
                    maxTemp = -(khgt * kwdt * kdpt);
                    for (uint16_t yy = 0; yy < pool; yy++) {
                        for (uint16_t xx = 0; xx < pool; xx++) {
                            outTemp = 0;
                            pWgt = pKrn + (k * pckWdt + ks) * (khgt * kwdt * kdpt) / pckWdt;
                            pIn = pAct + (y * pool + yy) * yCoeff + (x * pool + xx) * xCoeff;
                            // K-Y dim
                            for (uint16_t ky = 0; ky < khgt; ky++) {
                                // K-X dim
                                for (uint16_t kx = 0; kx < kwdt; kx++) {
                                    // Z dim
                                    for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                        // XNOR multiplication
                                        xnorTmp = ~(*pIn++ ^ *pWgt++);
                                        // popcount
                                        // Accumulation
                                        outTemp += popcount(xnorTmp);
                                    } // Z dim
                                } // K-X dim
                                pIn += (wdth - kwdt) * dpth / pckWdt;
                            } // K-Y dim
                            // Adjust the output value
                            outTemp = outTemp - (khgt * kwdt * kdpt - outTemp);
                            // Maxpool
                            if (outTemp > maxTemp) { maxTemp = outTemp; }
                        } // X-MP
                    } // Y-MP
                    // Batch normalize/ binarize
                    //goto end;
                    *pOut++ = (float)*gamma++ * (((bnPrec)maxTemp - *mean++) / (*var++)) + *beta++;
                }
            }
        }
    }
}

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding, pooling
 * Outer loop: XY, Pad: yes, Pool: yes BatchNorm: yes, SIMD: no  
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] pool - pooling size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdPlXnor(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype* __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype* __restrict thresh, pckDtype* sign){

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = khgt*kwdt*kdpt/pckWdt;
   uint16_t  kyCoeff = kwdt*kdpt/pckWdt;
   uint16_t  kxCoeff = kdpt/pckWdt;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   pckDtype  *signs = sign;
   bnDtype   *threshLoc = thresh;
   uint16_t  cntCoeff = khgt*kwdt*kdpt;
   // Starting indices for padding
   uint16_t  xStart, yStart = 0;
   // Ending indices for padding
   uint16_t  xEnd, yEnd = 0;
   // For maxpooling
   int32_t  maxTemp = 0;
   int16_t  oHgt = (hght-khgt+2*pad+1)/pool;
   int16_t  oWdt = (wdth-kwdt+2*pad+1)/pool;
   int16_t  knCoeff = knum/pckWdt;
   int16_t  pInStrd = (wdth-kwdt)*kxCoeff;

   // Divide the input into 5 regions - top, bottom, left, right, middle 
   // Middle has no padding

   // Middle - no padding
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= oHgt - 2*((pad+pool-1)/pool); y++) {
      //printf("Y: %d\n", y);
      // X dim
      // Set the output pointer
      // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
      // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
      // Offset to this row pad*knum/pckWdt
      pRes = pOut + y*oHgt*knCoeff + ((pad+pool-1)/pool)*knCoeff;
      //printf("%d %d %d\n", pOut, pRes, (y)*((hght-khgt+2*pad+1)/pool)*knum/pckWdt + ((pad+pool-1)/pool)*knum/pckWdt);
      for (uint16_t x = ((pad+pool-1)/pool); x <= oWdt - 2*((pad+pool-1)/pool); x++) {
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         //printf("X: %d\n", x);
         // Outer loop - kernels
         pWgt = pKrn;   
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knCoeff; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -cntCoeff;
               for (uint16_t yy = 0; yy < pool; yy++) {
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     pIn = pAct + (y*pool+yy - pad)*yCoeff + (x*pool+xx- pad)*xCoeff;
                     pWgt = pKrn + (k*pckWdt + ks)*kCoeff;
                     outTemp = 0;
                     // K-Y dim
                     for (uint16_t ky = 0; ky < khgt; ky++) {
                        // K-X dim
                        for (uint16_t kx = 0; kx < kyCoeff; kx++) {
                           // XNOR multiplication
                           xnorTmp = ~ ( *pIn++ ^ *pWgt++ ); 
                           outTemp += popcount(xnorTmp);
                        } // K-X dim
                        // Move the activation pointer one row down
                        pIn += pInStrd;
                     } // K-Y dim
                     outTemp = 2*outTemp - cntCoeff;
                     //printf("OT: %d\n", outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                     // Shift based on current kernel slice
                  } // X-MP
               } // Y-MP
               // Binarize
               maxTemp = (bnPrec) maxTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            //printf("%d\n", pRes);
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }

   //// Top
   pRes = pOut;
   // Y dim
   // We need to make sure there's enough lines to do pooling
   //for (uint16_t y = 0; y < pad; y++) {
   for (uint16_t y = 0; y < (pad+pool-1)/pool; y++) {
      // X dim
      for (uint16_t x = 0; x < oWdt; x++) {
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knCoeff; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(cntCoeff);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     outTemp = 0;
                     // Move the wieight pointer to the fisrt useful (non-padded) weight block
                     //pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt;
                     pWgt = pKrn + (k*pckWdt + ks)*kCoeff + yStart*kyCoeff+ xStart*kxCoeff;   
                     xyCount = 0;
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + ((y*pool+yy)+ky-pad)*yCoeff + ((x*pool+xx)+xStart-pad)*xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                              xyCount++;
                              // Z dim
                              for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                                 // XNOR multiplication
                                 xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                                 // popcount
                                 outTemp += popcount(xnorTmp);
                              } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt-xEnd+xStart)*kxCoeff; 
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     //printf("%d, ",outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = (bnPrec) maxTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
   
   // Bottom 
   // Move the ouput pointer
   pRes = pOut + ((hght-khgt+2*pad)/pool + 1 - ((pad+pool-1)/pool))*((wdth-kwdt+2*pad+1)/pool)*knum/pckWdt;
   // Y dim
   //for (uint16_t y = hght-khgt+((pad+pool-1)/pool)+1; y < hght-khgt+2*((pad+pool-1)/pool)+1; y++) {
   for (uint16_t y = (hght-khgt+2*pad)/pool + 1 - ((pad+pool-1)/pool); y < (hght-khgt+2*pad)/pool + 1; y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+2*pad)/pool +1; x++) {
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knCoeff; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(cntCoeff);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     // Move the wieight pointer to the fisrt useful (non-padded) weight block
                     //pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
                     pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
                     outTemp = 0;
                     xyCount = 0;
                     //printf("%d %d %d %d %d %d %d %d\n", y, yy, x, xx, yStart, yEnd, xStart, xEnd);
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        //pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                        pIn = pAct + ((y*pool+yy)+ky-pad)*yCoeff + ((x*pool+xx)+xStart-pad)*xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                              xyCount++;
                              // Z dim
                              for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                                 // XNOR multiplication
                                 xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                                 // popcount
                                 // Accumulation
                                 outTemp += popcount(xnorTmp);
                              } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = (bnPrec) maxTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
  
   //// Left 
   pRes = pOut + ((pad+pool-1)/pool)*(oWdt)*knCoeff;
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= oHgt - 2*((pad+pool-1)/pool); y++) {
      // X dim
      for (uint16_t x = 0; x < ((pad+pool-1)/pool); x++) {
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knCoeff; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(cntCoeff);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     outTemp = 0;
                     xyCount = 0;
                     // Move the wieight pointer to the fisrt useful (non-padded) weight block
                     //pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt;
                     pWgt = pKrn + (k*pckWdt + ks)*kCoeff + yStart*kyCoeff + xStart*kxCoeff;   
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + ((y*pool+yy)+ky-pad)*yCoeff + ((x*pool+xx)+xStart-pad)*xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                              xyCount++;
                              // Z dim
                              for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                                 // XNOR multiplication
                                 xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                                 // popcount
                                 // Accumulation
                                 outTemp += popcount(xnorTmp);
                              } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt-xEnd+xStart)*kxCoeff; 
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = (bnPrec) maxTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(oWdt)*knum/pckWdt;
   }

   // Right 
   pRes = pOut + ((pad+pool-1)/pool)*(oWdt)*knum/pckWdt + ((oWdt) - ((pad+pool-1)/pool))*knum/pckWdt;
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= oHgt - 2*((pad+pool-1)/pool); y++) {
      // X dim
      for (uint16_t x = (wdth-kwdt+2*pad)/pool + 1 - ((pad+pool-1)/pool); x < (wdth-kwdt+2*pad)/pool + 1; x++) {
      // Restart kernel bn pointer
      threshLoc = thresh;
      signs = sign;
      //for (uint16_t x = 0; x < (wdth-kwdt+2*pad)/pool +1; x++) {
      //for (uint16_t x = (wdth-kwdt+2*pad+1)/pool; x < wdth-kwdt+2*pad+1; x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knCoeff; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -(cntCoeff);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     // Move the wieight pointer to the fisrt useful (non-padded) weight block
                     //pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
                     pWgt = pKrn + (k*pckWdt + ks)*kCoeff + yStart*kyCoeff + xStart*kxCoeff;   
                     outTemp = 0;
                     xyCount = 0;
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        //pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                        pIn = pAct + ((y*pool+yy)+ky-pad)*yCoeff + ((x*pool+xx)+xStart-pad)*xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                           xyCount++;
                           // Z dim
                           for (uint16_t z = 0; z < dpth/pckWdt; z++) {
                              // XNOR multiplication
                              xnorTmp = ~ ( *pIn++ ^ *pWgt++);
                              // popcount
                              // Accumulation
                              outTemp += popcount(xnorTmp);
                           } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt-xEnd+xStart)*kxCoeff; 
                     } // K-Y dim
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Binarize
               maxTemp = (bnPrec) maxTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(oWdt)*knCoeff+ ((oWdt) - ((pad+pool-1)/pool))*knCoeff;
   }
}

void CnBnPdPlXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, 
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, 
    const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad, 
    const uint8_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta) {
    // Temporary variables
    pckDtype xnorTmp = 0;
    int32_t  outTemp = 0;
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
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    bnDtype* pRes = pOut;

    // Outer loop - kernels
    for (uint16_t k = 0; k < knum / pckWdt; k++) {
        // Packed slices
        for (uint16_t ks = 0; ks < pckWdt; ks++) {
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
                            outTemp = 0;
                            pWgt = pKrn + (k * pckWdt + ks) * (khgt * kwdt * kdpt) / pckWdt;
                            pIn = pAct + (y * pool + yy) * yCoeff + (x * pool + xx) * xCoeff;
                            // K-Y dim
                            for (uint16_t ky = 0; ky < khgt; ky++) {
                                // K-X dim
                                for (uint16_t kx = 0; kx < kwdt; kx++) {
                                    // Z dim
                                    for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                        // XNOR multiplication
                                        xnorTmp = ~(*pIn++ ^ *pWgt++);
                                        // popcount
                                        // Accumulation
                                        outTemp += popcount(xnorTmp);
                                    } // Z dim
                                } // K-X dim
                                pIn += (wdth - kwdt) * dpth / pckWdt;
                            } // K-Y dim
                            // Adjust the output value
                            outTemp = outTemp - (khgt * kwdt * kdpt - outTemp);
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
        }
    }

}

#ifdef NEON
/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding, pooling, NEON
 * Outer loop: XY, Pad: yes, Pool: yes BatchNorm: yes, SIMD: NEON (128)  
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] pool - pooling size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdPlXnorNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   uint8_t  yCoeff  = wdth*dpth/pckWdt;
   uint8_t  xCoeff  = dpth/pckWdt;
   uint8_t  kCoeff  = khgt*kwdt*kdpt/pckWdt;
   uint8_t  kyCoeff = kwdt*kdpt/pckWdt;
   uint8_t  kxCoeff = kdpt/pckWdt;
   // Starting indices for padding
   uint8_t  xStart, yStart = 0;
   uint8_t  xxStart, yyStart = 0;
   // Ending indices for padding
   uint8_t  xEnd, yEnd = 0;
   uint8_t  xxEnd, yyEnd = 0;
   // XY count for padding adjustment
   uint8_t  gyCount = 0;
   // For maxpooling
   int32_t  maxTemp = 0;
   pckDtype  *signs = sign;
   bnDtype   *threshLoc = thresh;
   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int64x2_t vecOut ;

   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+2*pad)/pool +1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; yyStart = 0; }
      if (y*pool > hght-khgt+pad-pool) { yEnd = y*pool -(hght-khgt+pad-pool) + 1; } else { yEnd = khgt; yyEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+2*pad)/pool +1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; xxStart = 0; }
         if (x*pool > wdth-kwdt+pad-pool) { xEnd = x*pool - (wdth-kwdt+pad-pool) + 1; } else { xEnd = kwdt; xxEnd = kwdt; }
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               maxTemp = -(khgt*kwdt*kdpt);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  if (yStart != 0) { yyStart = yStart-yy; };
                  if (y*pool+yEnd >= hght) { yyEnd = yEnd-yy; };
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     if (xStart != 0) { xxStart = xStart-xx; };
                     if (x*pool+xEnd >= wdth) { xxEnd = xEnd-xx; };
                     xyCount = 0;
                     vecOut[0] = 0;
                     vecOut[1] = 0;
                     // K-Y dim
                     for (uint16_t ky = yyStart; ky < yyEnd; ky++) {
                        // K-X dim
                        for (uint16_t kx = xxStart; kx < xxEnd; kx++) {
                              xyCount++;
                              // Z dim
                              for (uint16_t z = 0; z < dpth/128; z++) {
                                 // Load values
                                 vecAct = vld1q_s32(pAct + (y*pool+yy+ky-pad)*yCoeff + (x*pool+xx+kx-pad)*xCoeff + z*128/pckWdt);
                                 vecWgt = vld1q_s32(pKrn + (k*pckWdt+ks)*kCoeff +  ky*kyCoeff + kx*kxCoeff+ z*128/pckWdt);
                                 // XNOR
                                 vecAct = veorq_s32(vecAct, vecWgt);
                                 vecAct = vmvnq_s32(vecAct);
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
                              } // Z dim
                        } // K-X dim
                     } // K-Y dim
                     // Extract the output
                     outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Batch normalize/ binarize
               maxTemp = (bnPrec) maxTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            pOut[y*(((wdth-kwdt+2*pad)/pool)+1)*knum/pckWdt + x*knum/pckWdt + k] = pckTemp;
         }
      }
   }

}
#endif

