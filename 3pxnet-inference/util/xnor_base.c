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
 * \file      xnor_base.c
 * \brief     Support functions for binarized neural networks 
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */


#include "xnor_base.h"

/**
 * @details Pack a vector of inputs into binary containers (words) - pointer version
 * 
 * @param[in] pSrc - pointer to the source vector
 * @param[out] pDst - pointer to the destination vector
 * @param[in] numIn - length of the input vector
 * @param[in] threshold - binarization threshold
 */
void packBinThrsPtr(actDtype * restrict pSrc, pckDtype * restrict pDst, uint32_t numIn, int32_t threshold) {

   // Temporary input
   pckDtype actTemp;
   // For holding batches of input values 
   pckDtype pckTemp = 0;
   // Loop counter
   uint32_t blkCnt = numIn;
   // Packing/shifting  index
   uint8_t packIdx = pckWdt-1;

   while (blkCnt > 0) {
      // Load value
      actTemp = (pckDtype) *pSrc++;
      // Threshold
      actTemp = actTemp >= threshold;
      // Shift
      actTemp = actTemp << packIdx;
      // Pack
      pckTemp |= actTemp;
      // Full output block - write out
      if (packIdx == 0) {
         *pDst++ = pckTemp;
         pckTemp = 0;
         packIdx = pckWdt-1;
      }
      // Decrement packing index
      else {
         packIdx--;
      }
      // Decrement input counter
      blkCnt--;
      // Write the last block
      if (blkCnt == 0 && packIdx != pckWdt-1) {
        *pDst++ = pckTemp;
      }
   }
}

/**
 * @details Pack a vector of inputs into binary containers (words) - pointer version/ floating input
 * 
 * @param[in] pSrc - pointer to the source vector
 * @param[out] pDst - pointer to the destination vector
 * @param[in] numIn - length of the input vector
 * @param[in] threshold - binarization threshold
 */
void packBinThrsPtrFlt(float * restrict pSrc, pckDtype * restrict pDst, uint32_t numIn, float threshold) {

   // Temporary input
   float    actTemp;
   // Integer conversion
   pckDtype actInt;
   // For holding batches of input values 
   pckDtype pckTemp = 0;
   // Loop counter
   uint32_t blkCnt = numIn;
   // Packing/shifting  index
   uint8_t packIdx = pckWdt-1;

   while (blkCnt > 0) {
      // Load value
      actTemp = *pSrc++;
      // Threshold
      actInt  = actTemp >= threshold;
      // Shift
      actInt  = actInt  << packIdx;
      // Pack
      pckTemp |= actInt; 
      // Full output block - write out
      if (packIdx == 0) {
         *pDst++ = pckTemp;
         pckTemp = 0;
         packIdx = pckWdt-1;
      }
      // Decrement packing index
      else {
         packIdx--;
      }
      // Decrement input counter
      blkCnt--;
      // Write the last block
      if (blkCnt == 0 && packIdx != pckWdt-1) {
        *pDst++ = pckTemp;
      }
   }
}


/**
 * @details Pack a vector of inputs into binary containers (words) - array version
 * 
 * @param[in] pSrc - pointer to the source vector
 * @param[out] pDst - pointer to the destination vector
 * @param[in] numIn - length of the input vector
 * @param[in] threshold - binarization threshold
 */
void packBinThrsArr(uint8_t * restrict pSrc, pckDtype * restrict pDst, uint32_t numIn, int32_t threshold) {

   // Temporary input
   pckDtype actTemp;
   // For holding batches of input values 
   pckDtype pckTemp = 0;
   
   for (uint16_t blkCnt = 0; blkCnt < numIn/pckWdt; blkCnt++) {
      pckTemp = 0;
      for (uint8_t packIdx = 0; packIdx < pckWdt; packIdx++) {
         // Binarize
         actTemp = pSrc[blkCnt*pckWdt + packIdx] >= threshold;
         // Shift
         actTemp = actTemp << (pckWdt-1-packIdx);
         // Pack
         pckTemp |= actTemp;
      } 
      // Write output block
      pDst[blkCnt] = pckTemp; 
   }
}


/**
 * @details Permute inputs through matrix multiplication - pointer version
 * 
 * @param[in] pSrc - pointer to the source vector
 * @param[in] pPerm - pointer to the permutation matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[out] pDst - pointer to the destination vector
 */
void permInMatPtr(actDtype * restrict pSrc, uint8_t * restrict pPerm, uint16_t numIn, actDtype * restrict pDst) {
   
   // Temporary accumulation value
   actDtype temp;
   // Source vector pointer
   actDtype *pIn = pSrc;

   // Loop through outputs
   for (int i = 0; i< numIn; i++) {
      // Reset the accumulator and source pointer
      temp = 0;
      pIn = pSrc;
      // Loop through inputs
      for (int j=0; j<numIn; j++) {
         temp += (*pIn++) * (*pPerm++);   
      }
      // Write output out
      *pDst++ = temp;
   } 
} 

/**
 * @details Permute inputs through permutation indices - pointer version.
 * Note: Ideally this should be done in place to save buffer space.
 * This would require changing the order of indices.
 * 
 * @param[in] pSrc - pointer to the source vector
 * @param[in] pPermInd - pointer to the permutation vector (containing indices after permutation)
 * @param[in] numIn - length of the input vector
 * @param[out] pDst - pointer to the destination vector
 */
void permInIndPtr(float * restrict pSrc, uint16_t * restrict pPermInd, uint16_t numIn, float * restrict pDst) {

   for (int i = 0; i<numIn; i++) {
      *(pDst + *pPermInd++) = *pSrc++; 
   }
}

/**
 * @details Permute inputs through permutation indices - array version.
 * Note: Ideally this should be done in place to save buffer space.
 * This would require changing the order of indices.
 * 
 * @param[in] pSrc - pointer to the source vector
 * @param[in] pPermInd - pointer to the permutation vector (containing indices after permutation)
 * @param[in] numIn - length of the input vector
 * @param[out] pDst - pointer to the destination vector
 */
void permInIndArr(actDtype * restrict pSrc, uint16_t * restrict pPermInd, uint16_t numIn, actDtype * restrict pDst) {

   for (int i = 0; i<numIn; i++) {
      pDst[pPermInd[i]] = pSrc[i];
   }
}



