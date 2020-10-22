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
 * \file      3pxnet_cn.h
 * \brief     3PXNet convolutional layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#ifndef XNOR_SPARSEPK_CN_H
#define XNOR_SPARSEPK_CN_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "datatypes.h"
#ifdef NEON
#include "arm_neon.h"
#endif /* NEON */
#include <stdint.h>
#include <stdlib.h>
#include "utils.h"
/**
 * @brief 3PXNet binarized convolutional (Cn) layer with output binarization - general wrapper.
 */

uint8_t Cn3pxnWrap(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pIn, const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint16_t pad, const uint16_t pool, bnDtype * __restrict thresh, pckDtype * sign);
/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - general wrapper.
 */
uint8_t Cn3pxnNoBinWrap(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, uint8_t* __restrict pInd, const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint16_t pad, const uint16_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);
/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/no padding/ no pooling/.
 */
void Cn3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pInd, const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut);

/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/ no pooling/.
 */
void CnPd3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pIn, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad);

/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/pooling/.
 */
void CnPdPl3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pIn, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool);


/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/no padding/ no pooling/batch norm.
 */
void CnBn3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pIn, const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign);

/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/no padding/ no pooling/batch norm.
 */
void CnBn3pxnNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, uint8_t* __restrict pIn, const uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);
/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/ no pooling/batch norm.
 */
void CnBnPd3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pIn, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, bnDtype * __restrict thresh, pckDtype * __restrict sign);
/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/ no pooling/batch norm.
 */
void CnBnPd3pxnNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, uint8_t* __restrict pIn, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);
#ifdef NEON
/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/ no pooling/batch norm/NEON.
 */
void CnBnPd3pxnNeonQ4(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pIn, const uint8_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, bnDtype * __restrict thresh, pckDtype * __restrict sign);
#endif

/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/pooling/batch norm.
 */
void CnBnPdPl3pxn(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pIn, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype * __restrict thresh, pckDtype * __restrict sign);
/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/pooling/batch norm.
 */
void CnBnPdPl3pxnNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, uint8_t* __restrict pIn, uint16_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta);
#ifdef NEON
/**
 * @brief 3PXNet binarized convolutional (CN) layer with output binarization - array/padding/pooling/batch norm/NEON.
 */
void CnBnPdPl3pxnNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, uint8_t* __restrict pIn, const uint8_t kLen, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype * __restrict thresh, pckDtype * __restrict sign);
#endif



#ifdef __cplusplus
}
#endif

#endif /* XNOR_SPARSEPK_CN_H */

