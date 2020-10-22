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
 * \file      cn_reference.c
 * \brief     Reference Convolutional Layer for Binarized Implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#include "cn_reference.h"
/**
 * @details Reference Fully Connected Layer 
 * @param[in] pAct    - pointer to the packed activation vector (Y/X/Z - depth first)
 * @param[in] pKrn    - pointer to the packed weight vector (K/Y/X/Z - depth first)
 * @param[in] dpth    - Depth (Z)
 * @param[in] wdth    - Width (X)
 * @param[in] hght    - Height (Y)
 * @param[in] kwdt    - Kernel width (KX)
 * @param[in] khgt    - Kernel height (KY)
 * @param[in] knum    - # of kernels
 * @param[out] pOut   - pointer to the output vector (Y/X/Z - depth first)
 * @param[in] pool    - pooling size (stride assumed 1)
 * @param[in] pMean   - pointer to mean vector (if NULL, Batch Norm is skipped)
 * @param[in] pVar    - pointer to variance vector
 * @param[in] pGamma  - pointer to gamma vector
 * @param[in] pBeta   - pointer to beta vector
 * @param[in] epsilon - batch norm epsilon value
 */
void refCn(int16_t * pAct, int16_t * pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, float * pOut,
           const uint16_t pool, float * pMean, float * pVar, float * pGamma, float * pBeta, float epsilon) {

   // Local pointer to activations
   int16_t *pIn = pAct;
   // Local weight pointer
   int16_t *pWgt = pKrn;
   // Batch norm pointers
   float   *pMn = pMean;
   float   *pVr = pVar;
   float   *pGm = pGamma;
   float   *pBt = pBeta;
   // For maxpooling
   float   maxTemp = 0.0;
   // Temp output
   float   outTemp = 0.0;

   // Outer loop - Y
   for (uint16_t y = 0; y < ((hght-khgt+1)/pool); y++) {
      // X
      for (uint16_t x = 0; x < ((wdth-kwdt+1)/pool); x++) {
         // Starting kernel pointer (beginning of first kernel)
         pWgt = pKrn;
         // Starting batch norm pointers
         pMn = pMean;
         pVr = pVar;
         pGm = pGamma;
         pBt = pBeta;
         // Kernel
         for (uint16_t kn = 0; kn < knum; kn++) {
            // Mpool patches
            maxTemp = -(float)(khgt*kwdt*dpth);
            for (uint16_t yy = 0; yy < pool; yy++) {
               for (uint16_t xx = 0; xx < pool; xx++) {
                  // Clear output
                  outTemp = 0.0;
                  // Set starting pointer
                  pIn = pAct + (y*pool+yy)*wdth*dpth + (x*pool+xx)*dpth;
                  // Set the weight pointer
                  pWgt = pKrn + kn*khgt*kwdt*dpth;
                  // Kernel-Y
                  for (uint16_t ky = 0; ky < khgt; ky++) {
                     // Kernel-X
                     for (uint16_t kx = 0; kx < kwdt; kx++) {
                        // Z
                        for (uint16_t kz = 0; kz < dpth; kz++) {
                           outTemp += (*pIn++) * (*pWgt++);
                        }
                     } // KX
                     // Move to the next row
                     pIn += (wdth-kwdt)*dpth;
                  } // KY
                  // Running maxpool
                  if (outTemp > maxTemp) { maxTemp = outTemp; }
               }
            }
            // Optional Batch Norm - skip if pMean is NULL
            if (pMean) {
               maxTemp = ((*pGm++)*(((float)maxTemp - *pMn++)/(sqrt(*pVr++ + epsilon))) + *pBt++);
            }
            *pOut++ = maxTemp;
         } // Kernel
      } // X
   } // Y
}


