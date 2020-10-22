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
 * \file      xnor_base.h
 * \brief     Support functions for binarized neural networks 
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#ifndef XNOR_BASE_H
#define XNOR_BASE_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "datatypes.h"
#ifdef NEON
#include "arm_neon.h"
#endif
#include <stdint.h>
#include "utils.h"

/**
 * @brief Pack a vector of inputs into binary containers (words) - pointer version
 */
void packBinThrsPtr(actDtype * restrict pSrc, pckDtype * restrict pDst, uint32_t numIn, int32_t threshold);

/**
 * @brief Pack a vector of inputs into binary containers (words) - pointer version/ floating input
 */
void packBinThrsPtrFlt(float * restrict pSrc, pckDtype * restrict pDst, uint32_t numIn, float threshold);

/**
 * @brief Pack a vector of inputs into binary containers (words) - array version
 */
void packBinThrsArr(uint8_t * restrict pSrc, pckDtype * restrict pDst, uint32_t numIn, int32_t threshold);

/**
 * @brief Permute inputs through matrix multiplication - pointer version
 */
void permInMatPtr(actDtype * restrict pSrc, uint8_t * restrict pPerm, uint16_t numIn, actDtype * restrict pDst);

/**
 * @brief Permute inputs through permutation indices - pointer version
 */
void permInIndPtr(float * restrict pSrc, uint16_t * restrict pPermInd, uint16_t numIn, float * restrict pDst);

/**
 * @brief Permute inputs through permutation indices - array version
 */
void permInIndArr(actDtype * restrict pSrc, uint16_t * restrict pPermInd, uint16_t numIn, actDtype * restrict pDst);



#ifdef __cplusplus
}
#endif

#endif /* XNOR_BASE_H */

