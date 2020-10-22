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
* IMPlIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

/*!
 * \file      xnor_dense_cn.h
 * \brief     Dense binarized (XNOR) convolutional layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#ifndef XNOR_DENSE_CN_H
#define XNOR_DENSE_CN_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "datatypes.h"
#ifdef NEON
#include "arm_neon.h"
#endif /* NEON */
#include <stdint.h>
#include "utils.h"

/**
 * @brief Dense binarized Convolutional (CN) layer with output binarization - general wrapper.
 */
uint8_t CnXnorWrap(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint16_t pad, const uint16_t pool, bnDtype * __restrict thresh, pckDtype * sign);

uint8_t CnXnorNoBinWrap(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint16_t pad, const uint16_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);

/**
 * @brief  Dense binarized Convolutional (CN) layer with output binarization - XY outer loop
 */
void CnXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut);

/**
 * @brief  Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, padding
 */ 
void CnPdXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad);

/**
 * @brief  Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, pooling
 */
void CnPlXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pool);


/**
 * @brief  Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, padding/pooling
 */
void CnPdPlXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool);


/**
 * @brief Dense binarized Convolutional (CN) layer with output binarization - Kernel outer loop, batch norm
 */
void CnBnXnorKOut(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * sign);

void CnBnXnorKOutNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);

/**
 * @brief  Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm
 */ 
void CnBnXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * sign);
void CnBnXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);
/**
 * @brief Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding
 */ 
void CnBnPdXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, bnDtype * __restrict thresh, pckDtype * sign);
void CnBnPdXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad,bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);
#ifdef NEON
/**
 * @brief  Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding, NEON
 */ 
#endif

/**
 * @brief Dense binarized Convolutional (Cn) layer with output binarization - XY outer loop, batch norm, pooling
 */ 
void CnBnPlXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pool, bnDtype * __restrict thresh, pckDtype * sign);
void CnBnPlXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);

/**
 * @brief Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding, pooling
 */ 
void CnBnPdPlXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype * __restrict thresh, pckDtype * sign);
void CnBnPdPlXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);
#ifdef NEON
/**
 * @brief Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding, pooling, NEON
 */ 
void CnBnPdPlXnorNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype * __restrict thresh, pckDtype * __restrict sign);
#endif




#ifdef __cplusplus
}
#endif

#endif /* XNOR_DENSE_CN_H */

