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
 * \file      datatypes.h
 * \brief     Datatypes used in XNOR and 3PXNet libraries
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */


#ifndef DATATYPES_H
#define DATATYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/// Activation/intermediate value datatype
#define actDtype int16_t
/// Activation/intermediate value size
#define actWdt   16
    
/// Packed format datatype
#define pckDtype int32_t
/// Packed format width
#define pckWdt   32
    
/// Internal/output datatype
#define intDtype int16_t

/// Batch norm (mean/var/gamma/beta/epsilon) datatype
#define bnDtype float
/// Batch norm operation precision
#define bnPrec float

#ifdef __cplusplus
}
#endif

#endif /* DATATYPES_H */

