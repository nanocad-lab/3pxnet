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
 * \brief     Macros for XNOR and 3PXNet functions
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "datatypes.h"
#ifdef NEON
#include "arm_neon.h"
#endif
#include <stdint.h>

#define c1_32  0x55555555
#define c2_32  0x33333333
#define c4_32  0x0F0F0F0F
#define c8_32  0x00FF00FF
#define c16_32 0x0000FFFF
#define c1_64  0x5555555555555555
#define c2_64  0x3333333333333333
#define c4_64  0x0F0F0F0F0F0F0F0F
#define c8_64  0x00FF00FF00FF00FF
#define c16_64 0x0000FFFF0000FFFF
#define c32_64 0x00000000FFFFFFFF

// Popcounts
#ifdef GCC
/// Software macro for 32-bit popcount (GCC)
#define popcount32(a)  ({a -= (a >> 1) & c1_32; a = ((a >> 2) & c2_32) + (a & c2_32); a = (a + (a >> 4)) & c4_32; a *= 0x01010101; a = a >> 24;} )
/// Software macro for 64-bit popcount (GCC)
#define popcount64(a)  ({a -= (a >> 1) & c1_64; a = ((a >> 2) & c2_64) + (a & c2_64); a = (a + (a >> 4)) & c4_64; a *= 0x0101010101010101; a = a >> 56;} )
#endif
#ifdef ARMCC
/// Software macro for 32-bit popcount (ARMCC)
#define popcount32(a) a -= (a >> 1) & c1_32; a = ((a >> 2) & c2_32) + (a & c2_32); a = (a + (a >> 4)) & c4_32; a *= 0x01010101; a = a >> 24;
/// Software macro for 64-bit popcount (ARMCC)
#define popcount64(a) a -= (a >> 1) & c1_64; a = ((a >> 2) & c2_64) + (a & c2_64); a = (a + (a >> 4)) & c4_64; a *= 0x0101010101010101; a = a >> 56;
#endif

// Popcounts
// HW/Compiler Popcount - choose the correct __builtin_popcount
#ifndef PCNTSW
// 32-bit packs
#ifdef PCK32
// 32-bit popcount on 32-bit arch - long
#ifdef ARCH32
   #define popcount __builtin_popcountl        
#endif
#ifdef ARCH64
// 32-bit popcount on 64-bit arch - int 
   #define popcount __builtin_popcount
#endif
#endif
// 64-bit packs
// 64-bit popcount on 64-bit arch - longlong
#ifdef PCK64
   #define popcount __builtin_popcountll
#endif
#endif
// SW popcount
#ifdef PCNTSW
#ifdef PCK32
   #define popcount popcount32
#endif
#ifdef PCK64
   #define popcount popcount64
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif /* UTILS_H */

