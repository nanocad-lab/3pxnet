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
 * \file      fc_reference.c
 * \brief     Reference Fully-Connected Layer for Binarized Implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#include "fc_reference.h"

/**
 * @details Reference Fully Connected Layer 
 * @param[in] pAct    - pointer to the packed activation vector
 * @param[in] pWgt    - pointer to the packed weight vector
 * @param[in] numIn   - length of the input vector
 * @param[in] numOut  - length of the output vector
 * @param[out] pOut   - pointer to the packed output vector
 * @param[in] pMean   - pointer to mean vector (if NULL, Batch Norm is skipped)
 * @param[in] pVar    - pointer to variance vector
 * @param[in] pGamma  - pointer to gamma vector
 * @param[in] pBeta   - pointer to beta vector
 * @param[in] epsilon - batch norm epsilon value
 */
void refFc(int16_t * pAct, int16_t * pWgt, const uint16_t numIn, const uint16_t numOut, float * pOut,
           float * pMean, float * pVar, float * pGamma, float * pBeta, float epsilon) {
   
   // Local pointer to activations
   int16_t *pIn = pAct;

   // Outer loop - outputs
   for (uint16_t out = 0; out < numOut; out++) {
      // Reset activation pointer
      pIn = pAct;
      // Clear output
      *pOut = 0;
      // Inner loop - inputs
      for (uint16_t in = 0; in < numIn; in++) {
         // Multiply-Accumulate
         *pOut += (*pIn++) * (*pWgt++); 
      } // in
      if (pMean) {
         *pOut = ((*pGamma++)*(((float)*pOut - *pMean++)/(*pVar++ )) + *pBeta++);
      }
      // Optional Batch Norm - skip if pMean is NULL
      // Move onto the next output
      pOut++;
   } // out
}



