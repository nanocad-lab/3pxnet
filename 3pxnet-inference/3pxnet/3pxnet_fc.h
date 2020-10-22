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
 * \file      3pxnet_fc.h
 * \brief     3PXNet fully-connected layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#ifndef XNOR_SPARSEPK_FC_H
#define XNOR_SPARSEPK_FC_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "datatypes.h"
#ifdef NEON
#include "arm_neon.h"
#endif /* NEON */
#include <stdint.h>
#include <math.h>
#include "utils.h"

/**
 * @brief 3PXNet binarized Fully Connected (FC) layer with output binarization - general wrapper.
 */
uint8_t Fc3pxnWrap(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign);

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - general wrapper.
 */
uint8_t Fc3pxnNoBinWrap(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta);


/**
 * @brief  3PXNet binarized Fully Connected (FC) layer with output binarization - pointer version
 */
void Fc3pxnPtr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut);

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer with output binarization - array version
 */
void Fc3pxnArr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut);

#ifdef NEON

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer with output binarization - array version, NEON support
 */
void Fc3pxnNeon(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut);

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer with output binarization - array version, NEON support
 */
void Fc3pxnNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut);
#endif /* NEON */

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - pointer version
 */
void Fc3pxnPtrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut);

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - array version
 */
void Fc3pxnArrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut);

#ifdef NEON

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - array version, NEON support
 */
void Fc3pxnNeonNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut);

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - array version, NEON support
 */
void Fc3pxnNeonQNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut);
#endif /* NEON */

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer with output binarization - pointer version, batch norm
 */
void FcBn3pxnPtr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign);

/**
 * @brief 3PXNet binarized Fully Connected (FC) layer with output binarization - array version, batch norm
 */
void FcBn3pxnArr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign);

#ifdef NEON

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer with output binarization - array version, batch norm
 */
void FcBn3pxnNeon(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign);

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer with output binarization - array version, batch norm
 */
void FcBn3pxnNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign);
#endif /* NEON */

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - pointer version, batch norm
 */
void FcBn3pxnPtrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta);

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - array version, batch norm
 */
void FcBn3pxnArrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta);

#ifdef NEON

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - array version, batch norm, NEON support
 */
void FcBn3pxnNeonNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta);

/**
 * @brief  3PXNet binarized Fully Connected (FC) layer without output binarization - array version, batch norm, NEON support
 */
void FcBn3pxnNeonQNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta);
#endif /* NEON */




#ifdef __cplusplus
}
#endif

#endif /* XNOR_SPARSEPK_FC_H */

