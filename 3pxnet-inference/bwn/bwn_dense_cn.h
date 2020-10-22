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
 * \file      bwn_dense_cn.h
 * \brief     Dense binary-weight convolutional layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#ifndef BWN_DENSE_CNBN_H
#define BWN_DENSE_CNBN_H

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
 * @brief Dense binary-weight Convolutional (CN) layer with output binarization.
 */ 
void CnBnBwn(uint8_t * restrict pAct, int8_t * restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, const uint16_t pad, const uint16_t pool, pckDtype * restrict pOut, bnDtype * restrict thresh, pckDtype * sign);

#ifdef __cplusplus
}
#endif

#endif /* BWN_DENSE_CNBN_H */

