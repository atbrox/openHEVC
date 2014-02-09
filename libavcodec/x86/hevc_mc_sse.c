/*
 * Provide SSE MC functions for HEVC decoding
 * Copyright (c) 2013 Pierre-Edouard LEPERE
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */


#include "config.h"
#include "libavutil/avassert.h"
#include "libavutil/pixdesc.h"
#include "libavcodec/get_bits.h"
#include "libavcodec/hevc.h"
#include "libavcodec/x86/hevcdsp.h"

#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>

#define BIT_DEPTH 8

static const uint8_t epel_h_filter_shuffle_8[16] = {0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};

static const uint8_t epel_h_filter_shuffle1_10[16]= {0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9};
static const uint8_t epel_h_filter_shuffle2_10[16]= {4, 5, 6, 7, 8, 9,10,11, 6, 7, 8, 9,10,11,12,13};


static const uint8_t qpel_h_filter_shuffle_8[16] = {0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8};

static const uint8_t qpel_h_filter_1_8[16] = { -1, 4,-10, 58, 17, -5, 1, 0, -1,  4,-10, 58, 17, -5, 1,  0};

static const uint8_t qpel_h_filter_2_8[16] = { -1, 4,-11, 40, 40,-11, 4, -1, -1, 4,-11, 40, 40,-11, 4, -1};

static const uint8_t qpel_h_filter_3_8[16] = {  0, 1, -5, 17, 58,-10, 4, -1,  0, 1, -5, 17, 58,-10, 4, -1};

static const uint8_t qpel_hv_filter_1_8[16 * 4] = {
     -1,  4, -1,  4, -1,  4, -1,  4, -1,  4, -1,  4, -1,  4, -1,  4,
    -10, 58,-10, 58,-10, 58,-10, 58,-10, 58,-10, 58,-10, 58,-10, 58,
     17, -5, 17, -5, 17, -5, 17, -5, 17, -5, 17, -5, 17, -5, 17, -5,
      1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0
};

static const uint8_t qpel_hv_filter_2_8[16 * 4] = {
     -1,  4, -1,  4, -1,  4, -1,  4, -1,  4, -1,  4, -1,  4, -1,  4,
    -11, 40,-11, 40,-11, 40,-11, 40,-11, 40,-11, 40,-11, 40,-11, 40,
     40,-11, 40,-11, 40,-11, 40,-11, 40,-11, 40,-11, 40,-11, 40,-11,
      4, -1,  4, -1,  4, -1,  4, -1,  4, -1,  4, -1,  4, -1,  4, -1,
};

static const uint8_t qpel_hv_filter_3_8[16 * 4] = {
     0,   1,  0,   1,  0,   1,  0,   1,  0,   1,  0,   1,  0,   1,  0,   1,
    -5,  17, -5,  17, -5,  17, -5,  17, -5,  17, -5,  17, -5,  17, -5,  17,
    58, -10, 58, -10, 58, -10, 58, -10, 58, -10, 58, -10, 58, -10, 58, -10,
     4,  -1,  4,  -1,  4,  -1,  4,  -1,  4,  -1,  4,  -1,  4,  -1,  4,  -1
};


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
#define EPEL_V_FILTER(set)                                                     \
    const int8_t *filter_v = ff_hevc_epel_filters[my - 1];                     \
    const __m128i c1 = set(filter_v[0]);                                       \
    const __m128i c2 = set(filter_v[1]);                                       \
    const __m128i c3 = set(filter_v[2]);                                       \
    const __m128i c4 = set(filter_v[3])

#define EPEL_V_FILTER_8()                                                      \
    EPEL_V_FILTER(_mm_set1_epi16)
#define EPEL_V_FILTER_10()                                                     \
    EPEL_V_FILTER(_mm_set1_epi32)
#define EPEL_V_FILTER_14()  EPEL_V_FILTER_10()                                                    

#define EPEL_H_FILTER_8()                                                      \
    __m128i bshuffle1 = _mm_load_si128((__m128i *) epel_h_filter_shuffle_8); \
    __m128i r0        = _mm_loadl_epi64((__m128i *) &ff_hevc_epel_filters[mx - 1]);    \
    r0 = _mm_shuffle_epi32(r0, 0)
#define EPEL_H_FILTER_10()                                                     \
    __m128i bshuffle1 = _mm_load_si128((__m128i *) epel_h_filter_shuffle1_10); \
    __m128i bshuffle2 = _mm_load_si128((__m128i *) epel_h_filter_shuffle2_10); \
    __m128i r0        = _mm_loadu_si128((__m128i *) &ff_hevc_epel_filters[mx - 1]);\
    r0 = _mm_cvtepi8_epi16(r0);                                                \
    r0 = _mm_unpacklo_epi64(r0, r0)

#define QPEL_V_FILTER_1(inst)                                                  \
    const __m128i c1 = inst( -1);                                              \
    const __m128i c2 = inst(  4);                                              \
    const __m128i c3 = inst(-10);                                              \
    const __m128i c4 = inst( 58);                                              \
    const __m128i c5 = inst( 17);                                              \
    const __m128i c6 = inst( -5);                                              \
    const __m128i c7 = inst(  1);                                              \
    const __m128i c8 = _mm_setzero_si128()
#define QPEL_V_FILTER_2(inst)                                                  \
    const __m128i c1 = inst( -1);                                              \
    const __m128i c2 = inst(  4);                                              \
    const __m128i c3 = inst(-11);                                              \
    const __m128i c4 = inst( 40);                                              \
    const __m128i c5 = inst( 40);                                              \
    const __m128i c6 = inst(-11);                                              \
    const __m128i c7 = inst(  4);                                              \
    const __m128i c8 = inst( -1)
#define QPEL_V_FILTER_3(inst)                                                  \
    const __m128i c1 = _mm_setzero_si128();                                    \
    const __m128i c2 = inst(  1);                                              \
    const __m128i c3 = inst( -5);                                              \
    const __m128i c4 = inst( 17);                                              \
    const __m128i c5 = inst( 58);                                              \
    const __m128i c6 = inst(-10);                                              \
    const __m128i c7 = inst(  4);                                              \
    const __m128i c8 = inst( -1)

#define QPEL_V_FILTER_1_8()                                                    \
    QPEL_V_FILTER_1(_mm_set1_epi16)
#define QPEL_V_FILTER_2_8()                                                    \
    QPEL_V_FILTER_2(_mm_set1_epi16)
#define QPEL_V_FILTER_3_8()                                                    \
    QPEL_V_FILTER_3(_mm_set1_epi16)
#define QPEL_V_FILTER_1_10()                                                   \
    QPEL_V_FILTER_1(_mm_set1_epi32)
#define QPEL_V_FILTER_2_10()                                                   \
    QPEL_V_FILTER_2(_mm_set1_epi32)
#define QPEL_V_FILTER_3_10()                                                   \
    QPEL_V_FILTER_3(_mm_set1_epi32)

#define QPEL_V_FILTER_1_14() QPEL_V_FILTER_1_10()
#define QPEL_V_FILTER_2_14() QPEL_V_FILTER_2_10()
#define QPEL_V_FILTER_3_14() QPEL_V_FILTER_3_10()


#define QPEL_H_FILTER_1_8()                                                    \
    __m128i bshuffle1 = _mm_load_si128( (__m128i*) qpel_h_filter_shuffle_8); \
    __m128i r0 = _mm_load_si128( (__m128i*) qpel_h_filter_1_8)
#define QPEL_H_FILTER_2_8()                                                    \
    __m128i bshuffle1 = _mm_load_si128( (__m128i*) qpel_h_filter_shuffle_8); \
    __m128i r0 = _mm_load_si128( (__m128i*) qpel_h_filter_2_8)
#define QPEL_H_FILTER_3_8()                                                    \
    __m128i bshuffle1 = _mm_load_si128( (__m128i*) qpel_h_filter_shuffle_8); \
    __m128i r0 = _mm_load_si128( (__m128i*) qpel_h_filter_3_8)

#define QPEL_H_FILTER_1_10()                                                   \
    __m128i r0 = _mm_set_epi16(  0, 1, -5, 17, 58,-10,  4, -1)
#define QPEL_H_FILTER_2_10()                                                   \
    __m128i r0 = _mm_set_epi16( -1, 4,-11, 40, 40,-11,  4, -1)
#define QPEL_H_FILTER_3_10()                                                   \
    __m128i r0 = _mm_set_epi16( -1, 4,-10, 58, 17, -5,  1,  0)

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
#define SRC_INIT_8()                                                           \
    uint8_t  *src       = (uint8_t*) _src;                                     \
    ptrdiff_t srcstride = _srcstride
#define SRC_INIT_10()                                                          \
    uint16_t *src       = (uint16_t*) _src;                                    \
    ptrdiff_t srcstride = _srcstride >> 1
#define SRC_INIT_14() SRC_INIT_10()

#define SRC_INIT1_8()                                                          \
        src       = (uint8_t*) _src
#define SRC_INIT1_10()                                                         \
        src       = (uint16_t*) _src
#define SRC_INIT1_14() SRC_INIT1_10()

#define DST_INIT_8()                                                           \
    uint8_t  *dst       = (uint8_t*) _dst;                                     \
    ptrdiff_t dststride = _dststride
#define DST_INIT_10()                                                          \
    uint16_t *dst       = (uint16_t*) _dst;                                    \
    ptrdiff_t dststride = _dststride >> 1
#define DST_INIT_14() DST_INIT_10()

#define MC_LOAD_PIXEL()                                                        \
    x1 = _mm_loadu_si128((__m128i *) &src[x])

#define EPEL_H_LOAD2()                                                         \
    x1 = _mm_loadu_si128((__m128i *) &src[x - 1])
#define EPEL_H_LOAD4()                                                         \
    EPEL_H_LOAD2()
#define EPEL_H_LOAD8()                                                         \
    x1 = _mm_loadu_si128((__m128i *) &src[x - 1]);                             \
    x2 = _mm_srli_si128(x1, 4)
#define EPEL_H_LOAD16()                                                        \
    EPEL_H_LOAD8();                                                            \
    x3 = _mm_srli_si128(x1, 8);                                                \
    x4 = _mm_loadu_si128((__m128i *) &src[x + 11])
#define EPEL_H_LOAD32()                                                        \
    EPEL_H_LOAD16();                                                           \
    x5 = _mm_srli_si128(x4, 4);                                                \
    x6 = _mm_srli_si128(x4, 8);                                                \
    x7 = _mm_loadu_si128((__m128i *) &src[x + 23]);                            \
    x8 = _mm_srli_si128(x7, 4)

#define EPEL_V_LOAD(tab)                                                       \
    x1 = _mm_loadu_si128((__m128i *) &tab[x -     srcstride]);                 \
    x2 = _mm_loadu_si128((__m128i *) &tab[x                ]);                 \
    x3 = _mm_loadu_si128((__m128i *) &tab[x +     srcstride]);                 \
    x4 = _mm_loadu_si128((__m128i *) &tab[x + 2 * srcstride])

#define EPEL_V_LOAD3(tab)                                                       \
    x2 = _mm_loadu_si128((__m128i *) &tab[x -     srcstride]);                 \
    x3 = _mm_loadu_si128((__m128i *) &tab[x                ]);                 \
    x4 = _mm_loadu_si128((__m128i *) &tab[x +     srcstride])

#define QPEL_H_LOAD2()                                                         \
    x1 = _mm_loadu_si128((__m128i *) &src[x - 3]);                             \
    x2 = _mm_srli_si128(x1, 2)
#define QPEL_H_LOAD4()                                                         \
    QPEL_H_LOAD2()
#define QPEL_H_LOAD8()                                                         \
    QPEL_H_LOAD4();                                                            \
    x3 = _mm_srli_si128(x1, 4);                                                \
    x4 = _mm_srli_si128(x1, 6)

#define QPEL_V_LOAD(tab)                                                       \
    x1 = _mm_loadu_si128((__m128i *) &tab[x - 3 * srcstride]);                 \
    x2 = _mm_loadu_si128((__m128i *) &tab[x - 2 * srcstride]);                 \
    x3 = _mm_loadu_si128((__m128i *) &tab[x -     srcstride]);                 \
    x4 = _mm_loadu_si128((__m128i *) &tab[x                ]);                 \
    x5 = _mm_loadu_si128((__m128i *) &tab[x +     srcstride]);                 \
    x6 = _mm_loadu_si128((__m128i *) &tab[x + 2 * srcstride]);                 \
    x7 = _mm_loadu_si128((__m128i *) &tab[x + 3 * srcstride]);                 \
    x8 = _mm_loadu_si128((__m128i *) &tab[x + 4 * srcstride])

#define QPEL_V_LOAD_LO(tab)                                                       \
    x1 = _mm_loadu_si128((__m128i *) &tab[x - 3 * srcstride]);                 \
    x2 = _mm_loadu_si128((__m128i *) &tab[x - 2 * srcstride]);                 \
    x3 = _mm_loadu_si128((__m128i *) &tab[x -     srcstride]);                 \
    x4 = _mm_loadu_si128((__m128i *) &tab[x                ])

#define QPEL_V_LOAD_HI(tab)                                                       \
    x1 = _mm_loadu_si128((__m128i *) &tab[x +     srcstride]);                 \
    x2 = _mm_loadu_si128((__m128i *) &tab[x + 2 * srcstride]);                 \
    x3 = _mm_loadu_si128((__m128i *) &tab[x + 3 * srcstride]);                 \
    x4 = _mm_loadu_si128((__m128i *) &tab[x + 4 * srcstride])

#define PEL_STORE2(tab)                                                        \
    *((uint32_t *) &tab[x]) = _mm_cvtsi128_si32(r1)
#define PEL_STORE4(tab)                                                        \
    _mm_storel_epi64((__m128i *) &tab[x], r1)
#define PEL_STORE8(tab)                                                        \
    _mm_store_si128((__m128i *) &tab[x], r1)
#define PEL_STORE16(tab)                                                       \
    _mm_store_si128((__m128i *) &tab[x    ], r1);                             \
    _mm_store_si128((__m128i *) &tab[x + 8], r2)
#define PEL_STORE32(tab)                                                       \
    PEL_STORE16(tab);                                                          \
    _mm_store_si128((__m128i *) &tab[x +16], r3);                             \
    _mm_store_si128((__m128i *) &tab[x +24], r4)
////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

#define CVT4_8_16(dst, src)                                                    \
    dst ## 1 = _mm_cvtepu8_epi16(src ## 1);                                    \
    dst ## 2 = _mm_cvtepu8_epi16(src ## 2);                                    \
    dst ## 3 = _mm_cvtepu8_epi16(src ## 3);                                    \
    dst ## 4 = _mm_cvtepu8_epi16(src ## 4)

#define CVT4_16_32(dst, src)                                                   \
    dst ## 1 = _mm_cvtepi16_epi32(src ## 1);                                   \
    dst ## 2 = _mm_cvtepi16_epi32(src ## 2);                                   \
    dst ## 3 = _mm_cvtepi16_epi32(src ## 3);                                   \
    dst ## 4 = _mm_cvtepi16_epi32(src ## 4)

#define CVTHI4_16_32(dst, src)                                                 \
    dst ## 1 = _mm_unpackhi_epi64(src ## 1, c0);                               \
    dst ## 2 = _mm_unpackhi_epi64(src ## 2, c0);                               \
    dst ## 3 = _mm_unpackhi_epi64(src ## 3, c0);                               \
    dst ## 4 = _mm_unpackhi_epi64(src ## 4, c0);                               \
    CVT4_16_32(dst, dst)

#define MUL_ADD_H_1(mul, add, dst, src)                                        \
    src ## 1 = mul(src ## 1, r0);                                              \
    dst      = add(src ## 1, c0)

#define MUL_ADD_H_2_2(mul, add, dst, src)                                      \
    src ## 1 = mul(src ## 1, r0);                                              \
    src ## 2 = mul(src ## 2, r0);                                              \
    dst      = add(src ## 1, src ## 2)

#define MUL_ADD_H_4_2(mul, add, dst, src)                                      \
    MUL_ADD_H_2_2(mul, add, dst ## 1, src);                                    \
    src ## 3 = mul(src ## 3, r0);                                              \
    src ## 4 = mul(src ## 4, r0);                                              \
    dst ## 2 = add(src ## 3, src ## 4)

#define MUL_ADD_H_8_4(mul, add, dst, src)                                      \
    MUL_ADD_H_4_2(mul, add, dst , src);                                        \
    src ## 5 = mul(src ## 5, r0);                                              \
    src ## 6 = mul(src ## 6, r0);                                              \
    src ## 7 = mul(src ## 7, r0);                                              \
    src ## 8 = mul(src ## 8, r0);                                              \
    dst ## 3 = add(src ## 5, src ## 6);                                        \
    dst ## 4 = add(src ## 7, src ## 8)

#define MUL_ADD_H_2(mul, add, dst, src)                                        \
    MUL_ADD_H_2_2(mul, add, dst, src);                                         \
    dst      = add(dst, c0)

#define MUL_ADD_H_4(mul, add, dst, src)                                        \
    src ## 1 = mul(src ## 1, r0);                                              \
    src ## 2 = mul(src ## 2, r0);                                              \
    src ## 3 = mul(src ## 3, r0);                                              \
    src ## 4 = mul(src ## 4, r0);                                              \
    src ## 1 = add(src ## 1, src ## 2);                                        \
    src ## 3 = add(src ## 3, src ## 4);                                        \
    dst      = add(src ## 1, src ## 3)

#define MUL_ADD_V_4(mul, add, dst, src)                                        \
    dst = mul(src ## 1, c1);                                                   \
    dst = add(dst, mul(src ## 2, c2));                                         \
    dst = add(dst, mul(src ## 3, c3));                                         \
    dst = add(dst, mul(src ## 4, c4))
#define MUL_ADD_V_8(mul, add, dst, src)                                        \
    dst = mul(src ## 1, c1);                                                   \
    dst = add(dst, mul(src ## 2, c2));                                         \
    dst = add(dst, mul(src ## 3, c3));                                         \
    dst = add(dst, mul(src ## 4, c4));                                         \
    dst = add(dst, mul(src ## 5, c5));                                         \
    dst = add(dst, mul(src ## 6, c6));                                         \
    dst = add(dst, mul(src ## 7, c7));                                         \
    dst = add(dst, mul(src ## 8, c8))

#define MUL_ADD_V_LAST_4(mul, add, dst, src)                                        \
    dst = mul(src ## 1, c5);                                                   \
    dst = add(dst, mul(src ## 2, c6));                                         \
    dst = add(dst, mul(src ## 3, c7));                                         \
    dst = add(dst, mul(src ## 4, c8))

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
#define INST_SRC1_CST_2(inst, dst, src, cst)                                   \
    dst ## 1 = inst(src ## 1, cst);                                            \
    dst ## 2 = inst(src ## 2, cst)
#define INST_SRC1_CST_4(inst, dst, src, cst)                                   \
    INST_SRC1_CST_2(inst, dst, src, cst);                                      \
    dst ## 3 = inst(src ## 3, cst);                                            \
    dst ## 4 = inst(src ## 4, cst)
#define INST_SRC1_CST_8(inst, dst, src, cst)                                   \
    INST_SRC1_CST_4(inst, dst, src, cst);                                      \
    dst ## 5 = inst(src ## 5, cst);                                            \
    dst ## 6 = inst(src ## 6, cst);                                            \
    dst ## 7 = inst(src ## 7, cst);                                            \
    dst ## 8 = inst(src ## 8, cst)

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////
#define UNPACK_SRAI16_4(inst, dst, src)                                        \
    dst ## 1 = _mm_srai_epi32(inst(c0, src ## 1), 16);                         \
    dst ## 2 = _mm_srai_epi32(inst(c0, src ## 2), 16);                         \
    dst ## 3 = _mm_srai_epi32(inst(c0, src ## 3), 16);                         \
    dst ## 4 = _mm_srai_epi32(inst(c0, src ## 4), 16)
#define UNPACK_SRAI16_8(inst, dst, src)                                        \
    dst ## 1 = _mm_srai_epi32(inst(c0, src ## 1), 16);                         \
    dst ## 2 = _mm_srai_epi32(inst(c0, src ## 2), 16);                         \
    dst ## 3 = _mm_srai_epi32(inst(c0, src ## 3), 16);                         \
    dst ## 4 = _mm_srai_epi32(inst(c0, src ## 4), 16);                         \
    dst ## 5 = _mm_srai_epi32(inst(c0, src ## 5), 16);                         \
    dst ## 6 = _mm_srai_epi32(inst(c0, src ## 6), 16);                         \
    dst ## 7 = _mm_srai_epi32(inst(c0, src ## 7), 16);                         \
    dst ## 8 = _mm_srai_epi32(inst(c0, src ## 8), 16)

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

#define WEIGHTED_LOAD2()                                                       \
    r1 = _mm_loadl_epi64((__m128i *) &src[x    ])
#define WEIGHTED_LOAD4()                                                       \
    WEIGHTED_LOAD2()
#define WEIGHTED_LOAD8()                                                       \
    r1 = _mm_load_si128((__m128i *) &src[x    ])
#define WEIGHTED_LOAD16()                                                      \
    WEIGHTED_LOAD8();                                                          \
    r2 = _mm_load_si128((__m128i *) &src[x + 8])

#define WEIGHTED_LOAD2_1()                                                     \
    r3 = _mm_loadl_epi64((__m128i *) &src1[x    ])
#define WEIGHTED_LOAD4_1()                                                     \
    WEIGHTED_LOAD2_1()
#define WEIGHTED_LOAD8_1()                                                     \
    r3 = _mm_load_si128((__m128i *) &src1[x    ])
#define WEIGHTED_LOAD16_1()                                                    \
    WEIGHTED_LOAD8_1();                                                        \
    r4 = _mm_load_si128((__m128i *) &src1[x + 8])

#define WEIGHTED_STORE2_8()                                                      \
    r1 = _mm_packus_epi16(r1, r1);                                             \
    *((short *) (dst + x)) = _mm_extract_epi16(r1, 0)
#define WEIGHTED_STORE4_8()                                                    \
    r1 = _mm_packus_epi16(r1, r1);                                             \
    *((uint32_t *) &dst[x]) =_mm_cvtsi128_si32(r1)
#define WEIGHTED_STORE8_8()                                                      \
    r1 = _mm_packus_epi16(r1, r1);                                             \
    _mm_storel_epi64((__m128i *) &dst[x], r1)
#define WEIGHTED_STORE16_8()                                                   \
    r1 = _mm_packus_epi16(r1, r2);                                             \
    _mm_store_si128((__m128i *) &dst[x], r1)

#define WEIGHTED_STORE32_8()

#define WEIGHTED_STORE2_10() PEL_STORE2(dst);
#define WEIGHTED_STORE4_10() PEL_STORE4(dst);
#define WEIGHTED_STORE8_10() PEL_STORE8(dst);

#define WEIGHTED_STORE2_14() PEL_STORE2(dst);
#define WEIGHTED_STORE4_14() PEL_STORE4(dst);
#define WEIGHTED_STORE8_14() PEL_STORE8(dst);

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

#define WEIGHTED_INIT_0(H, D)                                                  \
    const int shift2 = 14 - D;                                                 \
    WEIGHTED_INIT_0_ ## D()

#define WEIGHTED_INIT_0_8()                                                    \
    const __m128i m1 = _mm_set1_epi16(1 << (14 - 8 - 1))

#define WEIGHTED_INIT_0_10()                                                    \
    const __m128i m1 = _mm_set1_epi16(1 << (14 - 10 - 1))

#define WEIGHTED_INIT_0_14()                                                    \
    const __m128i m1 = _mm_setzero_si128()

#define WEIGHTED_INIT_1(H, D)                                                  \
    const int shift2     = denom + 14 - D;                                     \
    const __m128i add   = _mm_set1_epi32(olxFlag << (D - 8));                  \
    const __m128i add2  = _mm_set1_epi32(1 << (shift2-1));                     \
    const __m128i m1    = _mm_set1_epi16(wlxFlag);                             \
    __m128i s1, s2, s3

#define WEIGHTED_INIT_2(H, D)                                                  \
    const int shift2 = 14 + 1 - D;                                             \
    const __m128i m1 = _mm_set1_epi16(1 << (14 - D))

#define WEIGHTED_INIT_3(H, D)                                                  \
    const int log2Wd = denom + 14 - D;                                         \
    const int shift2  = log2Wd + 1;                                            \
    const int o0     = olxFlag << (D - 8);                                     \
    const int o1     = ol1Flag << (D - 8);                                     \
    const __m128i m1 = _mm_set1_epi16(wlxFlag);                                \
    const __m128i m2 = _mm_set1_epi16(wl1Flag);                                \
    const __m128i m3 = _mm_set1_epi32((o0 + o1 + 1) << log2Wd);                \
    __m128i s1, s2, s3, s4, s5, s6


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

#define WEIGHTED_COMPUTE_0(reg1)                                               \
    reg1 = _mm_srai_epi16(_mm_adds_epi16(reg1, m1), shift2)
#define WEIGHTED_COMPUTE2_0()                                                  \
    WEIGHTED_COMPUTE_0(r1)
#define WEIGHTED_COMPUTE4_0()                                                  \
    WEIGHTED_COMPUTE_0(r1)
#define WEIGHTED_COMPUTE8_0()                                                  \
    WEIGHTED_COMPUTE_0(r1)
#define WEIGHTED_COMPUTE16_0()                                                 \
    WEIGHTED_COMPUTE_0(r1);                                                    \
    WEIGHTED_COMPUTE_0(r2)
#define WEIGHTED_COMPUTE32_0()


#define WEIGHTED_COMPUTE_1(reg1)                                               \
    s1   = _mm_mullo_epi16(reg1, m1);                                          \
    s2   = _mm_mulhi_epi16(reg1, m1);                                          \
    s3   = _mm_unpackhi_epi16(s1, s2);                                         \
    reg1 = _mm_unpacklo_epi16(s1, s2);                                         \
    reg1 = _mm_srai_epi32(_mm_add_epi32(reg1, add2), shift2);                  \
    s3   = _mm_srai_epi32(_mm_add_epi32(s3  , add2), shift2);                  \
    reg1 = _mm_add_epi32(reg1, add);                                           \
    s3   = _mm_add_epi32(s3  , add);                                           \
    reg1 = _mm_packus_epi32(reg1, s3)
#define WEIGHTED_COMPUTE2_1()                                                  \
    WEIGHTED_COMPUTE_1(r1)
#define WEIGHTED_COMPUTE4_1()                                                  \
    WEIGHTED_COMPUTE_1(r1)
#define WEIGHTED_COMPUTE4_1()                                                  \
    WEIGHTED_COMPUTE_1(r1)
#define WEIGHTED_COMPUTE8_1()                                                  \
    WEIGHTED_COMPUTE_1(r1)
#define WEIGHTED_COMPUTE16_1()                                                 \
    WEIGHTED_COMPUTE_1(r1);                                                    \
    WEIGHTED_COMPUTE_1(r2)
#define WEIGHTED_COMPUTE32_1()
#define WEIGHTED_COMPUTE32_2()
#define WEIGHTED_COMPUTE32_3()
#define WEIGHTED_COMPUTE_2(reg1, reg2)                                         \
    reg1 = _mm_adds_epi16(reg1, m1);                                           \
    reg1 = _mm_adds_epi16(reg1, reg2);                                         \
    reg2 = _mm_srai_epi16(reg1, shift2)
#define WEIGHTED_COMPUTE2_2()                                                  \
    WEIGHTED_LOAD2_1();                                                        \
    WEIGHTED_COMPUTE_2(r3, r1)
#define WEIGHTED_COMPUTE4_2()                                                  \
    WEIGHTED_LOAD4_1();                                                        \
    WEIGHTED_COMPUTE_2(r3, r1)
#define WEIGHTED_COMPUTE8_2()                                                  \
    WEIGHTED_LOAD8_1();                                                        \
    WEIGHTED_COMPUTE_2(r3, r1)
#define WEIGHTED_COMPUTE16_2()                                                 \
    WEIGHTED_LOAD16_1();                                                       \
    WEIGHTED_COMPUTE_2(r3, r1);                                                \
    WEIGHTED_COMPUTE_2(r4, r2)

#define WEIGHTED_COMPUTE_3(reg1, reg2)                                         \
    s1   = _mm_mullo_epi16(reg1, m1);                                          \
    s2   = _mm_mulhi_epi16(reg1, m1);                                          \
    s3   = _mm_mullo_epi16(reg2, m2);                                          \
    s4   = _mm_mulhi_epi16(reg2, m2);                                          \
    s5   = _mm_unpacklo_epi16(s1, s2);                                         \
    s6   = _mm_unpacklo_epi16(s3, s4);                                         \
    reg1 = _mm_unpackhi_epi16(s1, s2);                                         \
    reg2 = _mm_unpackhi_epi16(s3, s4);                                         \
    reg1 = _mm_add_epi32(reg1, reg2);                                          \
    reg2 = _mm_add_epi32(s5, s6);                                              \
    reg1 = _mm_srai_epi32(_mm_add_epi32(reg1, m3), shift2);                    \
    reg2 = _mm_srai_epi32(_mm_add_epi32(reg2, m3), shift2);                    \
    reg2 = _mm_packus_epi32(reg2, reg1)
#define WEIGHTED_COMPUTE2_3()                                                  \
    WEIGHTED_LOAD2_1();                                                        \
    WEIGHTED_COMPUTE_3(r3, r1)
#define WEIGHTED_COMPUTE4_3()                                                  \
    WEIGHTED_LOAD4_1();                                                        \
    WEIGHTED_COMPUTE_3(r3, r1)
#define WEIGHTED_COMPUTE8_3()                                                  \
    WEIGHTED_LOAD8_1();                                                        \
    WEIGHTED_COMPUTE_3(r3, r1)
#define WEIGHTED_COMPUTE16_3()                                                 \
    WEIGHTED_LOAD16_1();                                                       \
    WEIGHTED_COMPUTE_3(r3, r1);                                                \
    WEIGHTED_COMPUTE_3(r4, r2)

////////////////////////////////////////////////////////////////////////////////
// ff_hevc_put_hevc_mc_pixelsX_X_sse
////////////////////////////////////////////////////////////////////////////////
#define MC_PIXEL_COMPUTE2_8()                                                  \
    x2 = _mm_cvtepu8_epi16(x1);                                            \
    r1 = _mm_slli_epi16(x2, 14 - 8)
#define MC_PIXEL_COMPUTE4_8()                                                  \
    MC_PIXEL_COMPUTE2_8()
#define MC_PIXEL_COMPUTE8_8()                                                  \
    MC_PIXEL_COMPUTE2_8()
#define MC_PIXEL_COMPUTE16_8()                                                 \
    MC_PIXEL_COMPUTE2_8();                                                     \
    x3 = _mm_unpackhi_epi8(x1, c0);                                            \
    r2 = _mm_slli_epi16(x3, 14 - 8)

#define MC_PIXEL_COMPUTE2_10()                                                 \
    r1 = _mm_slli_epi16(x1, 14 - 10)
#define MC_PIXEL_COMPUTE4_10()                                                 \
    MC_PIXEL_COMPUTE2_10()
#define MC_PIXEL_COMPUTE8_10()                                                 \
    MC_PIXEL_COMPUTE2_10()

#define PUT_HEVC_EPEL_PIXELS(H, D)                                             \
void ff_hevc_put_hevc_epel_pixels ## H ## _ ## D ## _sse (                     \
                                   int16_t *dst, ptrdiff_t dststride,          \
                                   uint8_t *_src, ptrdiff_t _srcstride,        \
                                   int width, int height,                      \
                                   int mx, int my) {        \
    int x, y;                                                                  \
    __m128i x1, x2, x3, r1, r2;                                                \
    const __m128i c0    = _mm_setzero_si128();                                 \
    SRC_INIT_ ## D();                                                          \
    for (y = 0; y < height; y++) {                                             \
        for (x = 0; x < width; x += H) {                                       \
            MC_LOAD_PIXEL();                                                   \
            MC_PIXEL_COMPUTE ## H ## _ ## D();                                 \
            PEL_STORE ## H(dst);                                               \
        }                                                                      \
        src += srcstride;                                                      \
        dst += dststride;                                                      \
    }                                                                          \
}

#define PUT_HEVC_QPEL_PIXELS(H, D)                                             \
void ff_hevc_put_hevc_qpel_pixels ## H  ## _ ## D ## _sse (                    \
                                    int16_t *dst, ptrdiff_t dststride,         \
                                    uint8_t *_src, ptrdiff_t _srcstride,       \
                                    int width, int height) {                   \
    int x, y;                                                                  \
    __m128i x1, x2, x3, r1, r2;                                                \
    const __m128i c0    = _mm_setzero_si128();                                 \
    SRC_INIT_ ## D();                                                          \
    for (y = 0; y < height; y++) {                                             \
        for (x = 0; x < width; x += H) {                                       \
            MC_LOAD_PIXEL();                                                   \
            MC_PIXEL_COMPUTE ## H ## _ ## D();                                 \
            PEL_STORE ## H(dst);                                               \
        }                                                                      \
        src += srcstride;                                                      \
        dst += dststride;                                                      \
    }                                                                          \
}

////////////////////////////////////////////////////////////////////////////////
// ff_hevc_put_hevc_epel_hX_X_sse
////////////////////////////////////////////////////////////////////////////////
#define EPEL_H_COMPUTE2_8()                                                    \
    x1 = _mm_shuffle_epi8(x1, bshuffle1);                                      \
    MUL_ADD_H_1(_mm_maddubs_epi16, _mm_hadd_epi16, r1, x)
#define EPEL_H_COMPUTE4_8()                                                    \
    EPEL_H_COMPUTE2_8()
#define EPEL_H_COMPUTE8_8()                                                    \
    INST_SRC1_CST_2(_mm_shuffle_epi8, x, x , bshuffle1);                       \
    MUL_ADD_H_2_2(_mm_maddubs_epi16, _mm_hadd_epi16, r1, x)
#define EPEL_H_COMPUTE16_8()                                                    \
    INST_SRC1_CST_4(_mm_shuffle_epi8, x, x , bshuffle1);                       \
    MUL_ADD_H_4_2(_mm_maddubs_epi16, _mm_hadd_epi16, r, x)
#define EPEL_H_COMPUTE32_8()                                                    \
    INST_SRC1_CST_8(_mm_shuffle_epi8, x, x , bshuffle1);                       \
    MUL_ADD_H_8_4(_mm_maddubs_epi16, _mm_hadd_epi16, r, x)

#define EPEL_H_COMPUTE2_10()                                                   \
    x1 = _mm_shuffle_epi8(x1, bshuffle1);                                      \
    MUL_ADD_H_1(_mm_madd_epi16, _mm_hadd_epi32, r1, x);                        \
    r1 = _mm_srai_epi32(r1, 10 - 8);                                           \
    r1 = _mm_packs_epi32(r1, c0)
#define EPEL_H_COMPUTE4_10()                                                   \
    x2 = _mm_shuffle_epi8(x1, bshuffle2);                                      \
    x1 = _mm_shuffle_epi8(x1, bshuffle1);                                      \
    MUL_ADD_H_2_2(_mm_madd_epi16, _mm_hadd_epi32, r1, x);                      \
    r1 = _mm_srai_epi32(r1, 10 - 8);                                           \
    r1 = _mm_packs_epi32(r1, c0)

#define PUT_HEVC_EPEL_H(H, D)                                                  \
void ff_hevc_put_hevc_epel_h ## H ## _ ## D ## _sse (                          \
                                   int16_t *dst, ptrdiff_t dststride,          \
                                   uint8_t *_src, ptrdiff_t _srcstride,        \
                                   int width, int height,                      \
                                   int mx, int my) {                           \
    int x, y;                                                                  \
    __m128i x1, x2, x3, x4, x5, x6, x7, x8,  r1, r2, r3, r4;                   \
    const __m128i c0     = _mm_setzero_si128();                                \
    SRC_INIT_ ## D();                                                          \
    EPEL_H_FILTER_ ## D();                                                     \
    for (y = 0; y < height; y++) {                                             \
        for (x = 0; x < width; x += H) {                                       \
            EPEL_H_LOAD ## H();                                                \
            EPEL_H_COMPUTE ## H ## _ ## D();                                   \
            PEL_STORE ## H(dst);                                               \
        }                                                                      \
        src += srcstride;                                                      \
        dst += dststride;                                                      \
    }                                                                          \
}

////////////////////////////////////////////////////////////////////////////////
// ff_hevc_put_hevc_epel_vX_X_sse
////////////////////////////////////////////////////////////////////////////////
#define EPEL_V_COMPUTE2_8()                                                    \
    CVT4_8_16( x, x);                                                          \
    MUL_ADD_V_4(_mm_mullo_epi16, _mm_adds_epi16, r1, x)
#define EPEL_V_COMPUTE4_8()                                                    \
    EPEL_V_COMPUTE2_8()
#define EPEL_V_COMPUTE8_8()                                                    \
    EPEL_V_COMPUTE2_8()
#define EPEL_V_COMPUTE16_8()                                                   \
    INST_SRC1_CST_4(_mm_unpackhi_epi8, t, x , c0);                             \
    MUL_ADD_V_4(_mm_mullo_epi16, _mm_adds_epi16, r2, t);                       \
    EPEL_V_COMPUTE2_8()

#define EPEL_V_COMPUTE2_10()                                                   \
    CVT4_16_32(x, x);                                                          \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r1, x);                        \
    r1 = _mm_srai_epi32(r1, shift);                                            \
    r1 = _mm_packs_epi32(r1, c0)
#define EPEL_V_COMPUTE4_10()                                                   \
    EPEL_V_COMPUTE2_10()
#define EPEL_V_COMPUTE8_10()                                                   \
    CVT4_16_32( t, x);                                 \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r2, t);                        \
    CVTHI4_16_32( x, x);                                 \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r1, x);                        \
    r2 = _mm_srai_epi32(r2, shift);                                            \
    r1 = _mm_srai_epi32(r1, shift);                                            \
    r1 = _mm_packs_epi32(r2, r1)

#define EPEL_V_COMPUTE2_14()                                                   \
    CVT4_16_32(x, x);                                                          \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r1, x);                        \
    r1 = _mm_srai_epi32(r1, shift);                                            \
    r1 = _mm_packs_epi32(r1, c0)
#define EPEL_V_COMPUTE4_14()     EPEL_V_COMPUTE2_14()
#define EPEL_V_COMPUTE8_14()                                                   \
    CVT4_16_32(t, x);                                                          \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r2, t);                        \
    CVTHI4_16_32(x, x);                                                        \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r1, x);                        \
    r2 = _mm_srai_epi32(r2, shift);                                            \
    r1 = _mm_srai_epi32(r1, shift);                                            \
    r1 = _mm_packs_epi32(r2, r1)


#define EPEL_V_COMPUTEB2_8()                                                    \
    CVT4_8_16(y, x);                                                           \
    MUL_ADD_V_4(_mm_mullo_epi16, _mm_adds_epi16, r1, y)
#define EPEL_V_COMPUTEB4_8()                                                    \
    EPEL_V_COMPUTEB2_8()
#define EPEL_V_COMPUTEB8_8()                                                    \
    EPEL_V_COMPUTEB2_8()
#define EPEL_V_COMPUTEB16_8()                                                   \
    INST_SRC1_CST_4(_mm_unpackhi_epi8, t, x , c0);                             \
    MUL_ADD_V_4(_mm_mullo_epi16, _mm_adds_epi16, r2, t);                       \
    EPEL_V_COMPUTEB2_8()

#define EPEL_V_COMPUTEB2_10()                                                  \
    CVT4_16_32(y, x);                                                          \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r1, y);                        \
    r1 = _mm_srai_epi32(r1, shift);                                            \
    r1 = _mm_packs_epi32(r1, c0)
#define EPEL_V_COMPUTEB4_10()                                                  \
    EPEL_V_COMPUTEB2_10()
#define EPEL_V_COMPUTEB8_10()                                                  \
    CVT4_16_32(t, x);                                                          \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r2, t);                        \
    INST_SRC1_CST_4(_mm_unpackhi_epi16, y, x , c0);                            \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r1, y);                        \
    r2 = _mm_srai_epi32(r2, shift);                                            \
    r1 = _mm_srai_epi32(r1, shift);                                            \
    r1 = _mm_packs_epi32(r2, r1)

#define EPEL_V_COMPUTEB2_14()                                                   \
    CVT4_16_32(y, x);                                                          \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r1, y);                        \
    r1 = _mm_srai_epi32(r1, shift);                                            \
    r1 = _mm_packs_epi32(r1, c0)
#define EPEL_V_COMPUTEB4_14()                                                  \
    EPEL_V_COMPUTEB2_14()
#define EPEL_V_COMPUTEB8_14()                                                  \
    CVT4_16_32(t, x);                                                          \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r2, t);                        \
    CVTHI4_16_32(y, x);                                                        \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r1, y);                        \
    r2 = _mm_srai_epi32(r2, shift);                                            \
    r1 = _mm_srai_epi32(r1, shift);                                            \
    r1 = _mm_packs_epi32(r2, r1)


#define EPEL_V_SHIFT(tab)                                                      \
    x1= x2;                                                                    \
    x2= x3;                                                                    \
    x3= x4;                                                                    \
    x4 = _mm_loadu_si128((__m128i *) &tab[x + 2 * srcstride])
#if 1
#define PUT_HEVC_EPEL_V(V, D)                                                  \
void ff_hevc_put_hevc_epel_v ## V ## _ ## D ## _sse (                          \
                                   int16_t *dst, ptrdiff_t dststride,          \
                                   uint8_t *_src, ptrdiff_t _srcstride,        \
                                   int width, int height,                      \
                                   int mx, int my) {        \
    int x, y;                                                                  \
    int shift = D - 8;                                                         \
    __m128i x1, x2, x3, x4;                                                    \
    __m128i y1, y2, y3, y4;                                                    \
    __m128i t1, t2, t3, t4;                                                    \
    __m128i r1, r2;                                                            \
    const __m128i c0    = _mm_setzero_si128();                                 \
    SRC_INIT_ ## D();                                                          \
    uint16_t  *_dst       = (uint16_t*) dst;                                   \
    EPEL_V_FILTER_ ## D();                                                     \
                                                                               \
    for (x = 0; x < width; x+= V) {                                            \
    EPEL_V_LOAD3(src);                                                         \
        for(y = 0; y < height; y++) {                                          \
            EPEL_V_SHIFT(src);                                                 \
            EPEL_V_COMPUTEB ## V ## _ ## D();                                  \
            PEL_STORE ## V(_dst);                                              \
            src += srcstride;                                                  \
            _dst += dststride;                                                 \
        }                                                                      \
        SRC_INIT1_ ## D();                                                     \
        _dst       = (uint16_t*) dst;                                          \
    }                                                                          \
}
#else
#define PUT_HEVC_EPEL_V(V, D)                                                  \
void ff_hevc_put_hevc_epel_v ## V ## _ ## D ## _sse (                          \
                                   int16_t *dst, ptrdiff_t dststride,          \
                                   uint8_t *_src, ptrdiff_t _srcstride,        \
                                   int width, int height,                      \
                                   int mx, int my) {                           \
    int x, y;                                                                  \
    int shift = D - 8;                                                         \
    __m128i x1, x2, x3, x4;                                                    \
    __m128i y1, y2, y3, y4;                                                    \
    __m128i t1, t2, t3, t4;                                                    \
    __m128i r1, r2;                                                            \
    const __m128i c0  = _mm_setzero_si128();                                   \
    SRC_INIT_ ## D();                                                          \
    uint16_t  *_dst   = (uint16_t*) dst;                                       \
    EPEL_V_FILTER_ ## D();                                                     \
                                                                               \
    for (y = 0; y < height; y++) {                                             \
        for(x = 0; x < width; x += V) {                                        \
            EPEL_V_LOAD(src);                                                  \
            EPEL_V_COMPUTE ## V ## _ ## D();                                   \
            PEL_STORE ## V(dst);                                               \
        }                                                                      \
        src += srcstride;                                                      \
        dst += dststride;                                                      \
   }                                                                           \
}
#endif

////////////////////////////////////////////////////////////////////////////////
// ff_hevc_put_hevc_epel_hvX_X_sse
////////////////////////////////////////////////////////////////////////////////
#define PUT_HEVC_EPEL_HV(H, D)                                                 \
void ff_hevc_put_hevc_epel_hv ## H ## _ ## D ## _sse (                         \
                                   int16_t *dst, ptrdiff_t dststride,          \
                                   uint8_t *_src, ptrdiff_t _srcstride,        \
                                   int width, int height,                      \
                                   int mx, int my) {        \
                                                                        \
                                                                          \
}


////////////////////////////////////////////////////////////////////////////////
// ff_hevc_put_hevc_qpel_hX_X_X_sse
////////////////////////////////////////////////////////////////////////////////
#define QPEL_H_COMPUTE4_8()                                                    \
    INST_SRC1_CST_2(_mm_shuffle_epi8, x, x , bshuffle1);                       \
    MUL_ADD_H_2(_mm_maddubs_epi16, _mm_hadd_epi16, r1, x)
#define QPEL_H_COMPUTE8_8()                                                    \
    INST_SRC1_CST_4(_mm_shuffle_epi8, x, x , bshuffle1);                       \
    MUL_ADD_H_4(_mm_maddubs_epi16, _mm_hadd_epi16, r1, x)
#define QPEL_H_COMPUTE2_10()                                                   \
    MUL_ADD_H_2(_mm_madd_epi16, _mm_hadd_epi32, r1, x);                        \
    r1 = _mm_srai_epi32(r1, 10 - 8);                                           \
    r1 = _mm_packs_epi32(r1, c0)
#define QPEL_H_COMPUTE4_10()                                                   \
    QPEL_H_COMPUTE2_10()

#define PUT_HEVC_QPEL_H(H, F, D)                                               \
void ff_hevc_put_hevc_qpel_h ## H ## _ ## F ## _ ## D ## _sse (                \
                                    int16_t *dst, ptrdiff_t dststride,         \
                                    uint8_t *_src, ptrdiff_t _srcstride,       \
                                    int width, int height) {                   \
    int x, y;                                                                  \
    __m128i x1, x2, x3, x4, r1;                                                \
    const __m128i c0    = _mm_setzero_si128();                                 \
    SRC_INIT_ ## D();                                                          \
    QPEL_H_FILTER_ ## F ## _ ## D();                                           \
    for (y = 0; y < height; y++) {                                             \
        for (x = 0; x < width; x += H) {                                       \
            QPEL_H_LOAD ## H();                                                \
            QPEL_H_COMPUTE ## H ## _ ## D();                                   \
            PEL_STORE ## H(dst);                                               \
        }                                                                      \
        src += srcstride;                                                      \
        dst += dststride;                                                      \
    }                                                                          \
}

////////////////////////////////////////////////////////////////////////////////
// ff_hevc_put_hevc_qpel_vX_X_X_sse
////////////////////////////////////////////////////////////////////////////////

#define QPEL_V_COMPUTE_FIRST4_8()                                              \
    CVT4_8_16(x, x);                                                           \
    MUL_ADD_V_4(_mm_mullo_epi16, _mm_adds_epi16, r1, x)
#define QPEL_V_COMPUTE_FIRST8_8()                                              \
    QPEL_V_COMPUTE_FIRST4_8()
#define QPEL_V_COMPUTE_FIRST16_8()                                             \
    INST_SRC1_CST_4(_mm_unpackhi_epi8, t, x , c0);                             \
    MUL_ADD_V_4(_mm_mullo_epi16, _mm_adds_epi16, r2, t);                       \
    QPEL_V_COMPUTE_FIRST4_8()

#define QPEL_V_COMPUTE_LAST4_8()                                               \
    CVT4_8_16(x, x);                                                           \
    MUL_ADD_V_LAST_4(_mm_mullo_epi16, _mm_adds_epi16, r3, x)
#define QPEL_V_COMPUTE_LAST8_8()                                               \
    QPEL_V_COMPUTE_LAST4_8()
#define QPEL_V_COMPUTE_LAST16_8()                                              \
    INST_SRC1_CST_4(_mm_unpackhi_epi8, t, x , c0);                             \
    MUL_ADD_V_LAST_4(_mm_mullo_epi16, _mm_adds_epi16, r4, t);                  \
    QPEL_V_COMPUTE_LAST4_8()

#define QPEL_V_MERGE4_8()                                                      \
    r1= _mm_add_epi16(r1,r3)
#define QPEL_V_MERGE8_8()                                                      \
        QPEL_V_MERGE4_8()
#define QPEL_V_MERGE16_8()                                                     \
    r1= _mm_add_epi16(r1,r3);                                                  \
    r2= _mm_add_epi16(r2,r4)


#define QPEL_V_COMPUTE_FIRST2_10()                                             \
    CVT4_16_32(x, x);                                                          \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r1, x)
#define QPEL_V_COMPUTE_FIRST4_10()                                             \
    QPEL_V_COMPUTE_FIRST2_10()

#define QPEL_V_COMPUTE_FIRST8_10()                                             \
    CVTHI4_16_32(t, x);                                                        \
    MUL_ADD_V_4(_mm_mullo_epi32, _mm_add_epi32, r2, t);                        \
    QPEL_V_COMPUTE_FIRST2_10()


#define QPEL_V_COMPUTE_LAST2_10()                                              \
    CVT4_16_32(x, x);                                                          \
    MUL_ADD_V_LAST_4(_mm_mullo_epi32, _mm_add_epi32, r3, x)

#define QPEL_V_COMPUTE_LAST4_10()                                              \
    QPEL_V_COMPUTE_LAST2_10()

#define QPEL_V_COMPUTE_LAST8_10()                                              \
    CVTHI4_16_32(t, x);                                                        \
    MUL_ADD_V_LAST_4(_mm_mullo_epi32, _mm_add_epi32, r4, t);                   \
    QPEL_V_COMPUTE_LAST2_10()

#define QPEL_V_COMPUTE_FIRST2_14() QPEL_V_COMPUTE_FIRST2_10()
#define QPEL_V_COMPUTE_FIRST4_14() QPEL_V_COMPUTE_FIRST4_10()
#define QPEL_V_COMPUTE_FIRST8_14() QPEL_V_COMPUTE_FIRST8_10()

#define QPEL_V_COMPUTE_LAST2_14() QPEL_V_COMPUTE_LAST2_10()
#define QPEL_V_COMPUTE_LAST4_14() QPEL_V_COMPUTE_LAST4_10()
#define QPEL_V_COMPUTE_LAST8_14() QPEL_V_COMPUTE_LAST8_10()

#define QPEL_V_MERGE2_10()                                                     \
    r1= _mm_add_epi32(r1,r3);                                                  \
r1 = _mm_srai_epi32(r1, shift);                                                \
r1 = _mm_packs_epi32(r1, c0)
#define QPEL_V_MERGE4_10()                                                     \
        QPEL_V_MERGE2_10()
#define QPEL_V_MERGE8_10()                                                     \
    r1= _mm_add_epi32(r1,r3);                                                  \
    r3= _mm_add_epi32(r2,r4);                                                  \
    r1 = _mm_srai_epi32(r1, shift);                                            \
    r3 = _mm_srai_epi32(r3, shift);                                            \
    r1 = _mm_packs_epi32(r1, r3)


#define QPEL_V_MERGE2_14() QPEL_V_MERGE2_10()
#define QPEL_V_MERGE4_14() QPEL_V_MERGE4_10()
#define QPEL_V_MERGE8_14() QPEL_V_MERGE8_10()

#define PUT_HEVC_QPEL_V(V, F, D)                                               \
void ff_hevc_put_hevc_qpel_v ## V ##_ ## F ## _ ## D ## _sse (                 \
                                    int16_t *dst, ptrdiff_t dststride,         \
                                    uint8_t *_src, ptrdiff_t _srcstride,       \
                                    int width, int height) {                   \
    int x, y;                                                                  \
    int shift = D - 8;                                                         \
    __m128i x1, x2, x3, x4, r1, r2,r3,r4;                                      \
    __m128i t1, t2, t3, t4;                                                    \
    const __m128i c0    = _mm_setzero_si128();                                 \
    SRC_INIT_ ## D();                                                          \
    QPEL_V_FILTER_ ## F ## _ ## D();                                           \
    for (y = 0; y < height; y++) {                                             \
        for (x = 0; x < width; x += V) {                                       \
            QPEL_V_LOAD_LO(src);                                               \
            QPEL_V_COMPUTE_FIRST ## V ## _ ## D();                             \
            QPEL_V_LOAD_HI(src);                                               \
            QPEL_V_COMPUTE_LAST ## V ## _ ## D();                              \
            QPEL_V_MERGE ## V ## _ ## D();                                     \
            PEL_STORE ## V(dst);                                               \
        }                                                                      \
        src += srcstride;                                                      \
        dst += dststride;                                                      \
    }                                                                          \
}

////////////////////////////////////////////////////////////////////////////////
// ff_hevc_put_hevc_qpel_hX_X_vX_X_sse
////////////////////////////////////////////////////////////////////////////////
#define PUT_HEVC_QPEL_HV(H, FH, FV, D)                                         \
void ff_hevc_put_hevc_qpel_h ## H ## _ ## FH ## _v_ ## FV ##_ ## D ## _sse (   \
                                    int16_t *dst, ptrdiff_t dststride,         \
                                    uint8_t *_src, ptrdiff_t _srcstride,       \
                                    int width, int height) {                   \
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//PUT_UNWEIGHTED_PRED_END
#define WEIGHTED_END_0()

//PUT_WEIGHTED_PRED_END
#define WEIGHTED_END_1()

//PUT_WEIGHTED_PRED_ARG_END
#define WEIGHTED_END_2()                                                       \
    src1 += src1stride;

//WEIGHTED_PRED_ARG_END
#define WEIGHTED_END_3()                                                       \
    src1 += src1stride;


////////////////////////////////////////////////////////////////////////////////

#define PUT_UNWEIGHTED_PRED(H, D)                                              \
static void put_unweighted_pred ## H ## _ ## D ##_sse (                        \
                                       uint8_t *dst, ptrdiff_t dststride,      \
                                       int16_t *src, ptrdiff_t srcstride,      \
                                       int width, int height) {                \
    int x, y;                                                                  \
    __m128i r1, r2;                                                            \
    WEIGHTED_INIT_0(H, D);                                                     \
    for (y = 0; y < height; y++) {                                             \
        for (x = 0; x < width; x += H) {                                       \
            WEIGHTED_LOAD ## H();                                              \
            WEIGHTED_COMPUTE ## H ## _0();                                     \
            WEIGHTED_STORE ## H ## _ ## D();                                   \
        }                                                                      \
        dst += dststride;                                                      \
        src += srcstride;                                                      \
    }                                                                          \
}


PUT_UNWEIGHTED_PRED(2,  8)
PUT_UNWEIGHTED_PRED(4,  8)
PUT_UNWEIGHTED_PRED(8,  8)
PUT_UNWEIGHTED_PRED(16, 8)

void ff_hevc_put_unweighted_pred_8_sse(
                                       uint8_t *dst, ptrdiff_t dststride,
                                       int16_t *src, ptrdiff_t srcstride,
                                       int width, int height) {
    if(!(width & 15)) {
        put_unweighted_pred16_8_sse(dst, dststride, src, srcstride, width, height);
    } else if(!(width & 7)) {
        put_unweighted_pred8_8_sse(dst, dststride, src, srcstride, width, height);
    } else if(!(width & 3)) {
        put_unweighted_pred4_8_sse(dst, dststride, src, srcstride, width, height);
    } else {
        put_unweighted_pred2_8_sse(dst, dststride, src, srcstride, width, height);
    }
}

#define PUT_WEIGHTED_PRED_AVG(H, D)                                            \
static void put_weighted_pred_avg ## H ## _ ## D ##_sse(                              \
                                uint8_t *dst, ptrdiff_t dststride,             \
                                int16_t *src1, int16_t *src,                   \
                                ptrdiff_t srcstride,                           \
                                int width, int height) {                       \
    int x, y;                                                                  \
    __m128i r1, r2, r3, r4;                                                    \
    WEIGHTED_INIT_2(H, D);                                                     \
    for (y = 0; y < height; y++) {                                             \
        for (x = 0; x < width; x += H) {                                       \
            WEIGHTED_LOAD ## H();                                              \
            WEIGHTED_COMPUTE ## H ## _2();                                     \
            WEIGHTED_STORE ## H ## _ ## D();                                   \
        }                                                                      \
        dst  += dststride;                                                     \
        src  += srcstride;                                                     \
        src1 += srcstride;                                                     \
    }                                                                          \
}
PUT_WEIGHTED_PRED_AVG(2,  8)
PUT_WEIGHTED_PRED_AVG(4,  8)
PUT_WEIGHTED_PRED_AVG(8,  8)
PUT_WEIGHTED_PRED_AVG(16, 8)

void ff_hevc_put_weighted_pred_avg_8_sse(
                                        uint8_t *dst, ptrdiff_t dststride,
                                        int16_t *src1, int16_t *src2,
                                        ptrdiff_t srcstride,
                                        int width, int height) {
    if(!(width & 15))
        put_weighted_pred_avg16_8_sse(dst, dststride,
                src1, src2, srcstride, width, height);
    else if(!(width & 7))
        put_weighted_pred_avg8_8_sse(dst, dststride,
                src1, src2, srcstride, width, height);
    else if(!(width & 3))
        put_weighted_pred_avg4_8_sse(dst, dststride,
                src1, src2, srcstride, width, height);
    else
        put_weighted_pred_avg2_8_sse(dst, dststride,
                src1, src2, srcstride, width, height);
}

////////////////////////////////////////////////////////////////////////////////
// ff_hevc_weighted_pred_8_sse
////////////////////////////////////////////////////////////////////////////////

#define WEIGHTED_PRED(H, D)                                                    \
static void weighted_pred ## H ## _ ## D ##_sse(                                      \
                                    uint8_t denom,                             \
                                    int16_t wlxFlag, int16_t olxFlag,          \
                                    uint8_t *dst, ptrdiff_t dststride,         \
                                    int16_t *src, ptrdiff_t srcstride,         \
                                    int width, int height) {                   \
    int x, y;                                                                  \
    __m128i r1, r2;                                                            \
    WEIGHTED_INIT_1(H, D);                                             \
    for (y = 0; y < height; y++) {                                             \
        for (x = 0; x < width; x += H) {                                       \
            WEIGHTED_LOAD ## H();                                              \
            WEIGHTED_COMPUTE ## H ## _1();                                     \
            WEIGHTED_STORE ## H ## _ ## D();                                   \
        }                                                                      \
        dst += dststride;                                                      \
        src += srcstride;                                                      \
    }                                                                          \
}
WEIGHTED_PRED(2, 8)
WEIGHTED_PRED(4, 8)
WEIGHTED_PRED(8, 8)
WEIGHTED_PRED(16, 8)

void ff_hevc_weighted_pred_8_sse(
                                 uint8_t denom,
                                 int16_t wlxFlag, int16_t olxFlag,
                                 uint8_t *dst, ptrdiff_t dststride,
                                 int16_t *src, ptrdiff_t srcstride,
                                 int width, int height) {
    if(!(width & 15))
        weighted_pred16_8_sse(denom, wlxFlag, olxFlag,
                dst, dststride, src, srcstride, width, height);
    else if(!(width & 7))
        weighted_pred8_8_sse(denom, wlxFlag, olxFlag,
                dst, dststride, src, srcstride, width, height);
    else if(!(width & 3))
        weighted_pred4_8_sse(denom, wlxFlag, olxFlag,
                dst, dststride, src, srcstride, width, height);
    else
        weighted_pred2_8_sse(denom, wlxFlag, olxFlag,
                dst, dststride, src, srcstride, width, height);
}

#define WEIGHTED_PRED_AVG(H, D)                                                   \
static void weighted_pred_avg ## H ## _ ## D ##_sse(                                  \
                                    uint8_t denom,                             \
                                    int16_t wlxFlag, int16_t wl1Flag,          \
                                    int16_t olxFlag, int16_t ol1Flag,          \
                                    uint8_t *dst, ptrdiff_t dststride,         \
                                    int16_t *src1, int16_t *src,               \
                                    ptrdiff_t srcstride,                       \
                                    int width, int height) {                   \
    int x, y;                                                                  \
    __m128i r1, r2, r3, r4;                                                    \
    WEIGHTED_INIT_3(H, D);                                             \
    for (y = 0; y < height; y++) {                                             \
        for (x = 0; x < width; x += H) {                                       \
            WEIGHTED_LOAD ## H();                                              \
            WEIGHTED_COMPUTE ## H ## _3();                                     \
            WEIGHTED_STORE ## H ## _ ## D();                                   \
        }                                                                      \
        dst  += dststride;                                                     \
        src  += srcstride;                                                     \
        src1 += srcstride;                                                     \
    }                                                                          \
}
WEIGHTED_PRED_AVG(2, 8)
WEIGHTED_PRED_AVG(4, 8)
WEIGHTED_PRED_AVG(8, 8)
WEIGHTED_PRED_AVG(16, 8)

void ff_hevc_weighted_pred_avg_8_sse(
                                 uint8_t denom,
                                 int16_t wl0Flag, int16_t wl1Flag,
                                 int16_t ol0Flag, int16_t ol1Flag,
                                 uint8_t *dst, ptrdiff_t dststride,
                                 int16_t *src1, int16_t *src2,
                                 ptrdiff_t srcstride,
                                 int width, int height) {
    if(!(width & 15))
        weighted_pred_avg16_8_sse(denom, wl0Flag, wl1Flag, ol0Flag, ol1Flag,
                dst, dststride, src1, src2, srcstride, width, height);
    else if(!(width & 7))
        weighted_pred_avg8_8_sse(denom, wl0Flag, wl1Flag, ol0Flag, ol1Flag,
                dst, dststride, src1, src2, srcstride, width, height);
    else if(!(width & 3))
        weighted_pred_avg4_8_sse(denom, wl0Flag, wl1Flag, ol0Flag, ol1Flag,
                dst, dststride, src1, src2, srcstride, width, height);
    else
        weighted_pred_avg2_8_sse(denom, wl0Flag, wl1Flag, ol0Flag, ol1Flag,
                dst, dststride, src1, src2, srcstride, width, height);
}



////////////////////////////////////////////////////////////////////////////////
// ff_hevc_put_hevc_qpel_hX_X_vX_X_sse
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#undef EPEL_FUNC
#define EPEL_FUNC(FUNC, H, D)                                                  \
    FUNC(H, D)
#undef QPEL_FUNC
#define QPEL_FUNC(FUNC, H, F, D)                                               \
    FUNC(H, F, D)
#define QPEL_FUNC_HV(FUNC, H, FH, FV, D)                                       \
    FUNC(H, FH, FV, D)


// ff_hevc_put_hevc_mc_pixelsX_X_sse
EPEL_FUNC( PUT_HEVC_EPEL_PIXELS,  2, 8)
EPEL_FUNC( PUT_HEVC_EPEL_PIXELS,  4, 8)
EPEL_FUNC( PUT_HEVC_EPEL_PIXELS,  8, 8)
EPEL_FUNC( PUT_HEVC_EPEL_PIXELS, 16, 8)

EPEL_FUNC( PUT_HEVC_EPEL_PIXELS,  2, 10)
EPEL_FUNC( PUT_HEVC_EPEL_PIXELS,  4, 10)
EPEL_FUNC( PUT_HEVC_EPEL_PIXELS,  8, 10)

EPEL_FUNC( PUT_HEVC_QPEL_PIXELS,  4,  8)
EPEL_FUNC( PUT_HEVC_QPEL_PIXELS,  8,  8)
EPEL_FUNC( PUT_HEVC_QPEL_PIXELS, 16,  8)

EPEL_FUNC( PUT_HEVC_QPEL_PIXELS,  4, 10)
EPEL_FUNC( PUT_HEVC_QPEL_PIXELS,  8, 10)

// ff_hevc_put_hevc_epel_hX_X_sse
EPEL_FUNC( PUT_HEVC_EPEL_H,  2,  8)
EPEL_FUNC( PUT_HEVC_EPEL_H,  4,  8)
EPEL_FUNC( PUT_HEVC_EPEL_H,  8,  8)
EPEL_FUNC( PUT_HEVC_EPEL_H, 16,  8)
EPEL_FUNC( PUT_HEVC_EPEL_H, 32,  8)

EPEL_FUNC( PUT_HEVC_EPEL_H,  2, 10)
EPEL_FUNC( PUT_HEVC_EPEL_H,  4, 10)

// ff_hevc_put_hevc_epel_vX_X_sse
EPEL_FUNC( PUT_HEVC_EPEL_V,  2, 8)
EPEL_FUNC( PUT_HEVC_EPEL_V,  4, 8)
EPEL_FUNC( PUT_HEVC_EPEL_V,  8, 8)
EPEL_FUNC( PUT_HEVC_EPEL_V, 16, 8)

EPEL_FUNC( PUT_HEVC_EPEL_V,  2, 10)
EPEL_FUNC( PUT_HEVC_EPEL_V,  4, 10)
EPEL_FUNC( PUT_HEVC_EPEL_V,  8, 10)

EPEL_FUNC( PUT_HEVC_EPEL_V,  2, 14)
EPEL_FUNC( PUT_HEVC_EPEL_V,  4, 14)
EPEL_FUNC( PUT_HEVC_EPEL_V,  8, 14)
/*
PUT_HEVC_EPEL_V( 2, 14)
PUT_HEVC_EPEL_V( 4, 14)
PUT_HEVC_EPEL_V( 8, 14)
*/

//EPEL_FUNC( PUT_HEVC_EPEL_V,  4, 14)
//EPEL_FUNC( PUT_HEVC_EPEL_V,  8, 14)

// ff_hevc_put_hevc_epel_hvX_X_sse


// ff_hevc_put_hevc_qpel_hX_X_X_sse
QPEL_FUNC( PUT_HEVC_QPEL_H,  4, 1,  8)
QPEL_FUNC( PUT_HEVC_QPEL_H,  4, 2,  8)
QPEL_FUNC( PUT_HEVC_QPEL_H,  4, 3,  8)

QPEL_FUNC( PUT_HEVC_QPEL_H,  8, 1,  8)
QPEL_FUNC( PUT_HEVC_QPEL_H,  8, 2,  8)
QPEL_FUNC( PUT_HEVC_QPEL_H,  8, 3,  8)

QPEL_FUNC( PUT_HEVC_QPEL_H,  4, 1, 10)
//QPEL_FUNC( PUT_HEVC_QPEL_H,  2, 2, 10)
//QPEL_FUNC( PUT_HEVC_QPEL_H,  2, 3, 10)

// ff_hevc_put_hevc_qpel_vX_X_X_sse
QPEL_FUNC( PUT_HEVC_QPEL_V,  4, 1,  8)
QPEL_FUNC( PUT_HEVC_QPEL_V,  4, 2,  8)
QPEL_FUNC( PUT_HEVC_QPEL_V,  4, 3,  8)

QPEL_FUNC( PUT_HEVC_QPEL_V,  8, 1,  8)
QPEL_FUNC( PUT_HEVC_QPEL_V,  8, 2,  8)
QPEL_FUNC( PUT_HEVC_QPEL_V,  8, 3,  8)

QPEL_FUNC( PUT_HEVC_QPEL_V, 16, 1,  8)
QPEL_FUNC( PUT_HEVC_QPEL_V, 16, 2,  8)
QPEL_FUNC( PUT_HEVC_QPEL_V, 16, 3,  8)

PUT_HEVC_QPEL_V( 4, 1, 14)
PUT_HEVC_QPEL_V( 4, 2, 14)
PUT_HEVC_QPEL_V( 4, 3, 14)

PUT_HEVC_QPEL_V( 8, 1, 14)
PUT_HEVC_QPEL_V( 8, 2, 14)
PUT_HEVC_QPEL_V( 8, 3, 14)


QPEL_FUNC( PUT_HEVC_QPEL_V,  4, 1, 10)
//QPEL_FUNC( PUT_HEVC_QPEL_V,  4, 2, 10)
//QPEL_FUNC( PUT_HEVC_QPEL_V,  4, 3, 10)

// ff_hevc_put_hevc_qpel_hvX_X_X_sse


//QPEL_FUNC_HV( PUT_HEVC_QPEL_HV,  2, 1, 1, 10)
//QPEL_FUNC_HV( PUT_HEVC_QPEL_HV,  2, 1, 2, 10)
//QPEL_FUNC_HV( PUT_HEVC_QPEL_HV,  2, 1, 3, 10)
//QPEL_FUNC_HV( PUT_HEVC_QPEL_HV,  2, 2, 1, 10)
//QPEL_FUNC_HV( PUT_HEVC_QPEL_HV,  2, 2, 2, 10)
//QPEL_FUNC_HV( PUT_HEVC_QPEL_HV,  2, 2, 3, 10)
//QPEL_FUNC_HV( PUT_HEVC_QPEL_HV,  2, 3, 1, 10)
//QPEL_FUNC_HV( PUT_HEVC_QPEL_HV,  2, 3, 2, 10)
//QPEL_FUNC_HV( PUT_HEVC_QPEL_HV,  2, 3, 3, 10)

void ff_hevc_put_unweighted_pred_sse(uint8_t *_dst, ptrdiff_t _dststride,
        int16_t *src, ptrdiff_t srcstride, int width, int height) {
    int x, y;
    uint8_t *dst = (uint8_t*) _dst;
    ptrdiff_t dststride = _dststride / sizeof(uint8_t);
    __m128i r0, r1, f0;
    int shift = 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int16_t offset = 1 << (shift - 1);
#else
    int16_t offset = 0;

#endif
    f0 = _mm_set1_epi16(offset);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 16) {
            r0 = _mm_load_si128((__m128i *) &src[x]);

            r1 = _mm_load_si128((__m128i *) &src[x + 8]);
            r0 = _mm_adds_epi16(r0, f0);

            r1 = _mm_adds_epi16(r1, f0);
            r0 = _mm_srai_epi16(r0, shift);
            r1 = _mm_srai_epi16(r1, shift);
            r0 = _mm_packus_epi16(r0, r1);

            _mm_store_si128((__m128i *) &dst[x], r0);
        }
        dst += dststride;
        src += srcstride;
    }
}

void ff_hevc_put_weighted_pred_avg_sse(uint8_t *_dst, ptrdiff_t _dststride,
        int16_t *src1, int16_t *src2, ptrdiff_t srcstride, int width,
        int height) {
    int x, y;
    uint8_t *dst = (uint8_t*) _dst;
    ptrdiff_t dststride = _dststride / sizeof(uint8_t);
    __m128i r0, r1, f0, r2, r3;
    int shift = 14 + 1 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif
    f0 = _mm_set1_epi16(offset);
    for (y = 0; y < height; y++) {

        for (x = 0; x < width; x += 16) {
            r0 = _mm_load_si128((__m128i *) &src1[x]);
            r1 = _mm_load_si128((__m128i *) &src1[x + 8]);
            r2 = _mm_load_si128((__m128i *) &src2[x]);
            r3 = _mm_load_si128((__m128i *) &src2[x + 8]);

            r0 = _mm_adds_epi16(r0, f0);
            r1 = _mm_adds_epi16(r1, f0);
            r0 = _mm_adds_epi16(r0, r2);
            r1 = _mm_adds_epi16(r1, r3);
            r0 = _mm_srai_epi16(r0, shift);
            r1 = _mm_srai_epi16(r1, shift);
            r0 = _mm_packus_epi16(r0, r1);

            _mm_store_si128((__m128i *) (dst + x), r0);
        }
        dst += dststride;
        src1 += srcstride;
        src2 += srcstride;
    }
}


void ff_hevc_weighted_pred_sse(uint8_t denom, int16_t wlxFlag, int16_t olxFlag,
        uint8_t *_dst, ptrdiff_t _dststride, int16_t *src, ptrdiff_t srcstride,
        int width, int height) {

    int log2Wd;
    int x, y;

    uint8_t *dst = (uint8_t*) _dst;
    ptrdiff_t dststride = _dststride / sizeof(uint8_t);
    __m128i x0, x1, x2, x3, c0, add, add2;

    log2Wd = denom + 14 - BIT_DEPTH;

    add = _mm_set1_epi32(olxFlag * (1 << (BIT_DEPTH - 8)));
    add2 = _mm_set1_epi32(1 << (log2Wd - 1));
    c0 = _mm_set1_epi16(wlxFlag);
    if (log2Wd >= 1)
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 16) {
                x0 = _mm_load_si128((__m128i *) &src[x]);
                x2 = _mm_load_si128((__m128i *) &src[x + 8]);
                x1 = _mm_unpackhi_epi16(_mm_mullo_epi16(x0, c0),
                        _mm_mulhi_epi16(x0, c0));
                x3 = _mm_unpackhi_epi16(_mm_mullo_epi16(x2, c0),
                        _mm_mulhi_epi16(x2, c0));
                x0 = _mm_unpacklo_epi16(_mm_mullo_epi16(x0, c0),
                        _mm_mulhi_epi16(x0, c0));
                x2 = _mm_unpacklo_epi16(_mm_mullo_epi16(x2, c0),
                        _mm_mulhi_epi16(x2, c0));
                x0 = _mm_add_epi32(x0, add2);
                x1 = _mm_add_epi32(x1, add2);
                x2 = _mm_add_epi32(x2, add2);
                x3 = _mm_add_epi32(x3, add2);
                x0 = _mm_srai_epi32(x0, log2Wd);
                x1 = _mm_srai_epi32(x1, log2Wd);
                x2 = _mm_srai_epi32(x2, log2Wd);
                x3 = _mm_srai_epi32(x3, log2Wd);
                x0 = _mm_add_epi32(x0, add);
                x1 = _mm_add_epi32(x1, add);
                x2 = _mm_add_epi32(x2, add);
                x3 = _mm_add_epi32(x3, add);
                x0 = _mm_packus_epi32(x0, x1);
                x2 = _mm_packus_epi32(x2, x3);
                x0 = _mm_packus_epi16(x0, x2);

                _mm_store_si128((__m128i *) (dst + x), x0);

            }
            dst += dststride;
            src += srcstride;
        }
    else
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 16) {

                x0 = _mm_load_si128((__m128i *) &src[x]);
                x2 = _mm_load_si128((__m128i *) &src[x + 8]);
                x1 = _mm_unpackhi_epi16(_mm_mullo_epi16(x0, c0),
                        _mm_mulhi_epi16(x0, c0));
                x3 = _mm_unpackhi_epi16(_mm_mullo_epi16(x2, c0),
                        _mm_mulhi_epi16(x2, c0));
                x0 = _mm_unpacklo_epi16(_mm_mullo_epi16(x0, c0),
                        _mm_mulhi_epi16(x0, c0));
                x2 = _mm_unpacklo_epi16(_mm_mullo_epi16(x2, c0),
                        _mm_mulhi_epi16(x2, c0));

                x0 = _mm_add_epi32(x0, add2);
                x1 = _mm_add_epi32(x1, add2);
                x2 = _mm_add_epi32(x2, add2);
                x3 = _mm_add_epi32(x3, add2);

                x0 = _mm_packus_epi32(x0, x1);
                x2 = _mm_packus_epi32(x2, x3);
                x0 = _mm_packus_epi16(x0, x2);

                _mm_store_si128((__m128i *) (dst + x), x0);

            }
            dst += dststride;
            src += srcstride;
        }
}

void ff_hevc_weighted_pred_avg_sse(uint8_t denom, int16_t wl0Flag,
        int16_t wl1Flag, int16_t ol0Flag, int16_t ol1Flag, uint8_t *_dst,
        ptrdiff_t _dststride, int16_t *src1, int16_t *src2, ptrdiff_t srcstride,
        int width, int height) {
    int shift, shift2;
    int log2Wd;
    int o0;
    int o1;
    int x, y;
    uint8_t *dst = (uint8_t*) _dst;
    ptrdiff_t dststride = _dststride / sizeof(uint8_t);
    __m128i x0, x1, x2, x3, r0, r1, r2, r3, c0, c1, c2;
    shift = 14 - BIT_DEPTH;
    log2Wd = denom + shift;

    o0 = (ol0Flag) * (1 << (BIT_DEPTH - 8));
    o1 = (ol1Flag) * (1 << (BIT_DEPTH - 8));
    shift2 = (log2Wd + 1);
    c0 = _mm_set1_epi16(wl0Flag);
    c1 = _mm_set1_epi16(wl1Flag);
    c2 = _mm_set1_epi32((o0 + o1 + 1) << log2Wd);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 16) {
            x0 = _mm_load_si128((__m128i *) &src1[x]);
            x1 = _mm_load_si128((__m128i *) &src1[x + 8]);
            x2 = _mm_load_si128((__m128i *) &src2[x]);
            x3 = _mm_load_si128((__m128i *) &src2[x + 8]);

            r0 = _mm_unpacklo_epi16(_mm_mullo_epi16(x0, c0),
                    _mm_mulhi_epi16(x0, c0));
            r1 = _mm_unpacklo_epi16(_mm_mullo_epi16(x1, c0),
                    _mm_mulhi_epi16(x1, c0));
            r2 = _mm_unpacklo_epi16(_mm_mullo_epi16(x2, c1),
                    _mm_mulhi_epi16(x2, c1));
            r3 = _mm_unpacklo_epi16(_mm_mullo_epi16(x3, c1),
                    _mm_mulhi_epi16(x3, c1));
            x0 = _mm_unpackhi_epi16(_mm_mullo_epi16(x0, c0),
                    _mm_mulhi_epi16(x0, c0));
            x1 = _mm_unpackhi_epi16(_mm_mullo_epi16(x1, c0),
                    _mm_mulhi_epi16(x1, c0));
            x2 = _mm_unpackhi_epi16(_mm_mullo_epi16(x2, c1),
                    _mm_mulhi_epi16(x2, c1));
            x3 = _mm_unpackhi_epi16(_mm_mullo_epi16(x3, c1),
                    _mm_mulhi_epi16(x3, c1));
            r0 = _mm_add_epi32(r0, r2);
            r1 = _mm_add_epi32(r1, r3);
            r2 = _mm_add_epi32(x0, x2);
            r3 = _mm_add_epi32(x1, x3);

            r0 = _mm_add_epi32(r0, c2);
            r1 = _mm_add_epi32(r1, c2);
            r2 = _mm_add_epi32(r2, c2);
            r3 = _mm_add_epi32(r3, c2);

            r0 = _mm_srai_epi32(r0, shift2);
            r1 = _mm_srai_epi32(r1, shift2);
            r2 = _mm_srai_epi32(r2, shift2);
            r3 = _mm_srai_epi32(r3, shift2);

            r0 = _mm_packus_epi32(r0, r2);
            r1 = _mm_packus_epi32(r1, r3);
            r0 = _mm_packus_epi16(r0, r1);

            _mm_store_si128((__m128i *) (dst + x), r0);

        }
        dst += dststride;
        src1 += srcstride;
        src2 += srcstride;
    }
}

void ff_hevc_put_hevc_epel_pixels_8_sse(int16_t *dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
        ptrdiff_t my) {
    int x, y;
    __m128i x1, x2,x3;
    uint8_t *src = (uint8_t*) _src;
    if(!(width & 15)){
        x3= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 16) {

                x1 = _mm_loadu_si128((__m128i *) &src[x]);
                x2 = _mm_unpacklo_epi8(x1, x3);

                x1 = _mm_unpackhi_epi8(x1, x3);

                x2 = _mm_slli_epi16(x2, 6);
                x1 = _mm_slli_epi16(x1, 6);
                _mm_store_si128((__m128i *) &dst[x], x2);
                _mm_store_si128((__m128i *) &dst[x + 8], x1);

            }
            src += srcstride;
            dst += dststride;
        }
    }else  if(!(width & 7)){
        x1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 8) {

                x2 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_unpacklo_epi8(x2, x1);
                x2 = _mm_slli_epi16(x2, 6);
                _mm_store_si128((__m128i *) &dst[x], x2);

            }
            src += srcstride;
            dst += dststride;
        }
    }else  if(!(width & 3)){
        x1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 4) {

                x2 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_unpacklo_epi8(x2,x1);

                x2 = _mm_slli_epi16(x2, 6);

                _mm_storel_epi64((__m128i *) &dst[x], x2);

            }
            src += srcstride;
            dst += dststride;
        }
    }else{
        x1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 2) {

                x2 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_unpacklo_epi8(x2, x1);
                x2 = _mm_slli_epi16(x2, 6);
                *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(x2);

            }
            src += srcstride;
            dst += dststride;
        }
    }

}

void ff_hevc_put_hevc_epel_pixels_10_sse(int16_t *dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t _srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
        ptrdiff_t my) {
    int x, y;
    __m128i x1, x2,x3;
    uint16_t *src = (uint16_t*) _src;
    ptrdiff_t srcstride = _srcstride>>1;
    if(!(width & 7)){
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 8) {

                x2 = _mm_loadu_si128((__m128i *) &src[x]);
                x2 = _mm_slli_epi16(x2, 4);         //shift 14 - BIT LENGTH
                _mm_store_si128((__m128i *) &dst[x], x2);

            }
            src += srcstride;
            dst += dststride;
        }
    }else  if(!(width & 3)){
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 4) {

                x2 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_slli_epi16(x2, 4);     //shift 14 - BIT LENGTH

                _mm_storel_epi64((__m128i *) &dst[x], x2);

            }
            src += srcstride;
            dst += dststride;
        }
    }else{
        x1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 2) {

                x2 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_slli_epi16(x2, 4);     //shift 14 - BIT LENGTH
                *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(x2);

            }
            src += srcstride;
            dst += dststride;
        }
    }

}

void ff_hevc_put_hevc_epel_h_8_sse(int16_t *dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t _srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
        ptrdiff_t my) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride;
    const int8_t *filter = ff_hevc_epel_filters[mx - 1];
    __m128i r0, bshuffle1, bshuffle2, x1, x2, x3;
    r0= _mm_loadl_epi64(filter);
    r0= _mm_shuffle_epi32(r0,0);

    bshuffle1 = _mm_set_epi8(6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0);


    if(!(width & 7)){
        bshuffle2 = _mm_set_epi8(10, 9, 8, 7, 9, 8, 7, 6, 8, 7, 6, 5, 7, 6, 5,
                4);
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 8) {

                x1 = _mm_loadu_si128((__m128i *) &src[x - 1]);
                x2 = _mm_shuffle_epi8(x1, bshuffle1);
                x3 = _mm_shuffle_epi8(x1, bshuffle2);

                /*  PMADDUBSW then PMADDW     */
                x2 = _mm_maddubs_epi16(x2, r0);
                x3 = _mm_maddubs_epi16(x3, r0);
                x2 = _mm_hadd_epi16(x2, x3);
                _mm_store_si128((__m128i *) &dst[x], x2);
            }
            src += srcstride;
            dst += dststride;
        }
    }else if(!(width & 3)){

        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 4) {
                /* load data in register     */
                x1 = _mm_loadu_si128((__m128i *) &src[x-1]);
                x2 = _mm_shuffle_epi8(x1, bshuffle1);

                /*  PMADDUBSW then PMADDW     */
                x2 = _mm_maddubs_epi16(x2, r0);
                x2 = _mm_hadd_epi16(x2, _mm_setzero_si128());
                /* give results back            */
                _mm_storel_epi64((__m128i *) &dst[x], x2);
            }
            src += srcstride;
            dst += dststride;
        }
    }else{
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 2) {
                /* load data in register     */
                x1 = _mm_loadu_si128((__m128i *) &src[x-1]);
                x2 = _mm_shuffle_epi8(x1, bshuffle1);

                /*  PMADDUBSW then PMADDW     */
                x2 = _mm_maddubs_epi16(x2, r0);
                x2 = _mm_hadd_epi16(x2, _mm_setzero_si128());
                /* give results back            */
                *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(x2);

            }
            src += srcstride;
            dst += dststride;
        }
    }
}

void ff_hevc_put_hevc_epel_h_10_sse(int16_t *dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t _srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
        ptrdiff_t my) {
    int x, y;
    uint16_t *src = (uint16_t*) _src;
    ptrdiff_t srcstride = _srcstride>>1;
    const int8_t *filter = ff_hevc_epel_filters[mx - 1];
    __m128i r0, bshuffle1, bshuffle2, x1, x2, x3, r1;
    int8_t filter_0 = filter[0];
    int8_t filter_1 = filter[1];
    int8_t filter_2 = filter[2];
    int8_t filter_3 = filter[3];
    r0 = _mm_set_epi16(filter_3, filter_2, filter_1,
            filter_0, filter_3, filter_2, filter_1, filter_0);
    bshuffle1 = _mm_set_epi8(9,8,7,6,5,4, 3, 2,7,6,5,4, 3, 2, 1, 0);

    if(!(width & 3)){
        bshuffle2 = _mm_set_epi8(13,12,11,10,9,8,7,6,11,10, 9,8,7,6,5, 4);
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 4) {

                x1 = _mm_loadu_si128((__m128i *) &src[x-1]);
                x2 = _mm_shuffle_epi8(x1, bshuffle1);
                x3 = _mm_shuffle_epi8(x1, bshuffle2);


                x2 = _mm_madd_epi16(x2, r0);
                x3 = _mm_madd_epi16(x3, r0);
                x2 = _mm_hadd_epi32(x2, x3);
                x2= _mm_srai_epi32(x2,2);   //>> (BIT_DEPTH - 8)

                x2 = _mm_packs_epi32(x2,r0);
                //give results back
                _mm_storel_epi64((__m128i *) &dst[x], x2);
            }
            src += srcstride;
            dst += dststride;
        }
    }else{
        r1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 2) {
                /* load data in register     */
                x1 = _mm_loadu_si128((__m128i *) &src[x-1]);
                x2 = _mm_shuffle_epi8(x1, bshuffle1);

                /*  PMADDUBSW then PMADDW     */
                x2 = _mm_madd_epi16(x2, r0);
                x2 = _mm_hadd_epi32(x2, r1);
                x2= _mm_srai_epi32(x2,2);   //>> (BIT_DEPTH - 8)
                x2 = _mm_packs_epi32(x2, r1);
                /* give results back            */
                *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(x2);

            }
            src += srcstride;
            dst += dststride;
        }
    }
}


void ff_hevc_put_hevc_epel_v_8_sse(int16_t *dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t _srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
        ptrdiff_t my) {
    int x, y;
    __m128i x0, x1, x2, x3, t0, t1, t2, t3, r0, f0, f1, f2, f3, r1;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    const int8_t *filter = ff_hevc_epel_filters[my - 1];
    int8_t filter_0 = filter[0];
    int8_t filter_1 = filter[1];
    int8_t filter_2 = filter[2];
    int8_t filter_3 = filter[3];
    f0 = _mm_set1_epi16(filter_0);
    f1 = _mm_set1_epi16(filter_1);
    f2 = _mm_set1_epi16(filter_2);
    f3 = _mm_set1_epi16(filter_3);

    if(!(width & 15)){
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 16) {
                /* check if memory needs to be reloaded */

                x0 = _mm_loadu_si128((__m128i *) &src[x - srcstride]);
                x1 = _mm_loadu_si128((__m128i *) &src[x]);
                x2 = _mm_loadu_si128((__m128i *) &src[x + srcstride]);
                x3 = _mm_loadu_si128((__m128i *) &src[x + 2 * srcstride]);

                t0 = _mm_unpacklo_epi8(x0, _mm_setzero_si128());
                t1 = _mm_unpacklo_epi8(x1, _mm_setzero_si128());
                t2 = _mm_unpacklo_epi8(x2, _mm_setzero_si128());
                t3 = _mm_unpacklo_epi8(x3, _mm_setzero_si128());

                x0 = _mm_unpackhi_epi8(x0, _mm_setzero_si128());
                x1 = _mm_unpackhi_epi8(x1, _mm_setzero_si128());
                x2 = _mm_unpackhi_epi8(x2, _mm_setzero_si128());
                x3 = _mm_unpackhi_epi8(x3, _mm_setzero_si128());

                /* multiply by correct value : */
                r0 = _mm_mullo_epi16(t0, f0);
                r1 = _mm_mullo_epi16(x0, f0);
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t1, f1));
                r1 = _mm_adds_epi16(r1, _mm_mullo_epi16(x1, f1));
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t2, f2));
                r1 = _mm_adds_epi16(r1, _mm_mullo_epi16(x2, f2));
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t3, f3));
                r1 = _mm_adds_epi16(r1, _mm_mullo_epi16(x3, f3));
                /* give results back            */
                _mm_store_si128((__m128i *) &dst[x], r0);
                _mm_store_si128((__m128i *) &dst[x + 8], r1);
            }
            src += srcstride;
            dst += dststride;
        }
    }else if(!(width & 7)){
        r1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 8){
                x0 = _mm_loadl_epi64((__m128i *) &src[x - srcstride]);
                x1 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_loadl_epi64((__m128i *) &src[x + srcstride]);
                x3 = _mm_loadl_epi64((__m128i *) &src[x + 2 * srcstride]);

                t0 = _mm_unpacklo_epi8(x0, r1);
                t1 = _mm_unpacklo_epi8(x1, r1);
                t2 = _mm_unpacklo_epi8(x2, r1);
                t3 = _mm_unpacklo_epi8(x3, r1);


                /* multiply by correct value : */
                r0 = _mm_mullo_epi16(t0, f0);
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t1, f1));
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t2, f2));
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t3, f3));
                /* give results back            */
                _mm_store_si128((__m128i *) &dst[x], r0);
            }
            src += srcstride;
            dst += dststride;
        }
    }else if(!(width & 3)){
        r1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for(x=0;x<width;x+=4){
                x0 = _mm_loadl_epi64((__m128i *) &src[x - srcstride]);
                x1 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_loadl_epi64((__m128i *) &src[x + srcstride]);
                x3 = _mm_loadl_epi64((__m128i *) &src[x + 2 * srcstride]);

                t0 = _mm_unpacklo_epi8(x0, r1);
                t1 = _mm_unpacklo_epi8(x1, r1);
                t2 = _mm_unpacklo_epi8(x2, r1);
                t3 = _mm_unpacklo_epi8(x3, r1);


                /* multiply by correct value : */
                r0 = _mm_mullo_epi16(t0, f0);
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t1, f1));
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t2, f2));
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t3, f3));
                /* give results back            */
                _mm_storel_epi64((__m128i *) &dst[x], r0);
            }
            src += srcstride;
            dst += dststride;
        }
    }else{
        r1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for(x=0;x<width;x+=2){
                x0 = _mm_loadl_epi64((__m128i *) &src[x - srcstride]);
                x1 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_loadl_epi64((__m128i *) &src[x + srcstride]);
                x3 = _mm_loadl_epi64((__m128i *) &src[x + 2 * srcstride]);

                t0 = _mm_unpacklo_epi8(x0, r1);
                t1 = _mm_unpacklo_epi8(x1, r1);
                t2 = _mm_unpacklo_epi8(x2, r1);
                t3 = _mm_unpacklo_epi8(x3, r1);


                /* multiply by correct value : */
                r0 = _mm_mullo_epi16(t0, f0);
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t1, f1));
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t2, f2));
                r0 = _mm_adds_epi16(r0, _mm_mullo_epi16(t3, f3));
                /* give results back            */
                *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(r0);


            }
            src += srcstride;
            dst += dststride;
        }
    }
}

void ff_hevc_put_hevc_epel_v_10_sse(int16_t *dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t _srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
        ptrdiff_t my, int16_t* mcbuffer) {
    int x, y;
    __m128i x0, x1, x2, x3, t0, t1, t2, t3, r0, f0, f1, f2, f3, r1, r2, r3;
    uint16_t *src = (uint16_t*) _src;
    ptrdiff_t srcstride = _srcstride >>1;
    const int8_t *filter = ff_hevc_epel_filters[my - 1];
    int8_t filter_0 = filter[0];
    int8_t filter_1 = filter[1];
    int8_t filter_2 = filter[2];
    int8_t filter_3 = filter[3];
    f0 = _mm_set1_epi16(filter_0);
    f1 = _mm_set1_epi16(filter_1);
    f2 = _mm_set1_epi16(filter_2);
    f3 = _mm_set1_epi16(filter_3);

    if(!(width & 7)){
        r1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for(x=0;x<width;x+=8){
                x0 = _mm_loadu_si128((__m128i *) &src[x - srcstride]);
                x1 = _mm_loadu_si128((__m128i *) &src[x]);
                x2 = _mm_loadu_si128((__m128i *) &src[x + srcstride]);
                x3 = _mm_loadu_si128((__m128i *) &src[x + 2 * srcstride]);

                // multiply by correct value :
                r0 = _mm_mullo_epi16(x0, f0);
                t0 = _mm_mulhi_epi16(x0, f0);

                x0= _mm_unpacklo_epi16(r0,t0);
                t0= _mm_unpackhi_epi16(r0,t0);

                r1 = _mm_mullo_epi16(x1, f1);
                t1 = _mm_mulhi_epi16(x1, f1);

                x1= _mm_unpacklo_epi16(r1,t1);
                t1= _mm_unpackhi_epi16(r1,t1);


                r2 = _mm_mullo_epi16(x2, f2);
                t2 = _mm_mulhi_epi16(x2, f2);

                x2= _mm_unpacklo_epi16(r2,t2);
                t2= _mm_unpackhi_epi16(r2,t2);


                r3 = _mm_mullo_epi16(x3, f3);
                t3 = _mm_mulhi_epi16(x3, f3);

                x3= _mm_unpacklo_epi16(r3,t3);
                t3= _mm_unpackhi_epi16(r3,t3);


                r0= _mm_add_epi32(x0,x1);
                r1= _mm_add_epi32(x2,x3);

                t0= _mm_add_epi32(t0,t1);
                t1= _mm_add_epi32(t2,t3);

                r0= _mm_add_epi32(r0,r1);
                t0= _mm_add_epi32(t0,t1);

                r0= _mm_srai_epi32(r0,2);//>> (BIT_DEPTH - 8)
                t0= _mm_srai_epi32(t0,2);//>> (BIT_DEPTH - 8)

                r0= _mm_packs_epi32(r0, t0);
                // give results back
                _mm_store_si128((__m128i *) &dst[x], r0);
            }
            src += srcstride;
            dst += dststride;
        }
    }else if(!(width & 3)){
        r1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for(x=0;x<width;x+=4){
                x0 = _mm_loadl_epi64((__m128i *) &src[x - srcstride]);
                x1 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_loadl_epi64((__m128i *) &src[x + srcstride]);
                x3 = _mm_loadl_epi64((__m128i *) &src[x + 2 * srcstride]);

                /* multiply by correct value : */
                r0 = _mm_mullo_epi16(x0, f0);
                t0 = _mm_mulhi_epi16(x0, f0);

                x0= _mm_unpacklo_epi16(r0,t0);

                r1 = _mm_mullo_epi16(x1, f1);
                t1 = _mm_mulhi_epi16(x1, f1);

                x1= _mm_unpacklo_epi16(r1,t1);


                r2 = _mm_mullo_epi16(x2, f2);
                t2 = _mm_mulhi_epi16(x2, f2);

                x2= _mm_unpacklo_epi16(r2,t2);


                r3 = _mm_mullo_epi16(x3, f3);
                t3 = _mm_mulhi_epi16(x3, f3);

                x3= _mm_unpacklo_epi16(r3,t3);


                r0= _mm_add_epi32(x0,x1);
                r1= _mm_add_epi32(x2,x3);
                r0= _mm_add_epi32(r0,r1);
                r0= _mm_srai_epi32(r0,2);//>> (BIT_DEPTH - 8)

                r0= _mm_packs_epi32(r0, r0);

                // give results back
                _mm_storel_epi64((__m128i *) &dst[x], r0);
            }
            src += srcstride;
            dst += dststride;
        }
    }else{
        r1= _mm_setzero_si128();
        for (y = 0; y < height; y++) {
            for(x=0;x<width;x+=2){
                x0 = _mm_loadl_epi64((__m128i *) &src[x - srcstride]);
                x1 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_loadl_epi64((__m128i *) &src[x + srcstride]);
                x3 = _mm_loadl_epi64((__m128i *) &src[x + 2 * srcstride]);

                /* multiply by correct value : */
                r0 = _mm_mullo_epi16(x0, f0);
                t0 = _mm_mulhi_epi16(x0, f0);

                x0= _mm_unpacklo_epi16(r0,t0);

                r1 = _mm_mullo_epi16(x1, f1);
                t1 = _mm_mulhi_epi16(x1, f1);

                x1= _mm_unpacklo_epi16(r1,t1);

                r2 = _mm_mullo_epi16(x2, f2);
                t2 = _mm_mulhi_epi16(x2, f2);

                x2= _mm_unpacklo_epi16(r2,t2);

                r3 = _mm_mullo_epi16(x3, f3);
                t3 = _mm_mulhi_epi16(x3, f3);

                x3= _mm_unpacklo_epi16(r3,t3);

                r0= _mm_add_epi32(x0,x1);
                r1= _mm_add_epi32(x2,x3);
                r0= _mm_add_epi32(r0,r1);
                r0= _mm_srai_epi32(r0,2);//>> (BIT_DEPTH - 8)

                r0= _mm_packs_epi32(r0, r0);

                /* give results back            */
                *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(r0);


            }
            src += srcstride;
            dst += dststride;
        }
    }
}


void ff_hevc_put_hevc_epel_hv_8_8_sse(int16_t *_dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t _srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
        ptrdiff_t my) {

    int x, y;
    uint8_t *src = (uint8_t*) _src - 1;
    int16_t *dst = _dst;
    ptrdiff_t srcstride = _srcstride;
    const int8_t *filter_h = ff_hevc_epel_filters[mx - 1];
    const int8_t *filter_v = ff_hevc_epel_filters[my - 1];
    __m128i r0, bshuffle1, bshuffle2, x0, x1, x2, x3, t0, t1, t2, t3, f0, f1,
    f2, f3, r1, r2, r3;
    __m128i xx0, xx1, xx2, xx3;
    int16_t mcbuffer[(MAX_PB_SIZE + 3) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    r0 = _mm_set1_epi32(*(uint32_t*) filter_h);
    bshuffle1 = _mm_set_epi8(6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0);

    src -= EPEL_EXTRA_BEFORE * srcstride;

    for (y = 0; y < height + EPEL_EXTRA; y++) {
        for (x = 0; x < width; x += 8) {

            x1 = _mm_loadu_si128((__m128i *) &src[x]);
            x2 = _mm_shuffle_epi8(x1, bshuffle1);
            x1 = _mm_loadu_si128((__m128i *) &src[x + 4]);
            x3 = _mm_shuffle_epi8(x1, bshuffle1);

            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x2 = _mm_hadd_epi16(x2, x3);
            _mm_store_si128((__m128i *) &tmp[x], x2);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer;

    // vertical treatment
    f3 = _mm_set1_epi16(filter_v[3]);
    f1 = _mm_set1_epi16(filter_v[1]);
    f2 = _mm_set1_epi16(filter_v[2]);
    f0 = _mm_set1_epi16(filter_v[0]);

    for (x = 0; x < width; x += 8) {
        for (y = 0; y < height; y++) {
            x0 = _mm_load_si128((__m128i *) &tmp[x]);
            x1 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);

            r0 = _mm_mullo_epi16(x0, f0);
            r1 = _mm_mulhi_epi16(x0, f0);
            r2 = _mm_mullo_epi16(x1, f1);
            r3 = _mm_mulhi_epi16(x1, f1);

            t0 = _mm_unpacklo_epi16(r0, r1);
            xx0 = _mm_unpackhi_epi16(r0, r1);

            t1 = _mm_unpacklo_epi16(r2, r3);
            xx1 = _mm_unpackhi_epi16(r2, r3);

            x2 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            r0 = _mm_mullo_epi16(x2, f2);
            r1 = _mm_mulhi_epi16(x2, f2);
            r2 = _mm_mullo_epi16(x3, f3);
            r3 = _mm_mulhi_epi16(x3, f3);

            t2 = _mm_unpacklo_epi16(r0, r1);
            xx2 = _mm_unpackhi_epi16(r0, r1);
            t3 = _mm_unpacklo_epi16(r2, r3);
            xx3 = _mm_unpackhi_epi16(r2, r3);

            r0 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(t0, t1), _mm_add_epi32(t2, t3)), 6);
            r1 = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(xx0, xx1), _mm_add_epi32(xx2, xx3)), 6);

            // give results back
            r0 = _mm_packs_epi32(r0, r1);
            _mm_store_si128((__m128i *) &dst[x], r0);
            tmp += MAX_PB_SIZE;
            dst += dststride;
        }
        tmp = mcbuffer;
        dst = _dst;
    }
}

void ff_hevc_put_hevc_epel_hv_8_4_sse(int16_t *_dst, ptrdiff_t dststride,
                                    uint8_t *_src, ptrdiff_t _srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
                                    ptrdiff_t my) {

    int x, y;
    uint8_t *src = (uint8_t*) _src - 1;
    int16_t *dst= _dst;
    ptrdiff_t srcstride = _srcstride;
    const int8_t *filter_h = ff_hevc_epel_filters[mx - 1];
    const int8_t *filter_v = ff_hevc_epel_filters[my - 1];
    __m128i r0, bshuffle1, x0, x1, x2, x3, f0, f1, f2, f3, r1;
    __m128i xx0, xx1, xx2, xx3;
    int16_t mcbuffer[(MAX_PB_SIZE + 3) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    r0 = _mm_set1_epi32(*(uint32_t*) filter_h);
    bshuffle1 = _mm_set_epi8(6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0);

    src -= EPEL_EXTRA_BEFORE * srcstride;

    for (y = 0; y < height + EPEL_EXTRA; y ++) {
        for(x=0;x<width;x+=4){
            // load data in register
            x1 = _mm_loadl_epi64((__m128i *) &src[x]);
            x1 = _mm_shuffle_epi8(x1, bshuffle1);
            //  PMADDUBSW then PMADDW
            x1 = _mm_maddubs_epi16(x1, r0);
            x1 = _mm_hadd_epi16(x1, _mm_setzero_si128());

            // give results back
            _mm_storel_epi64((__m128i *) &tmp[x], x1);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer;

    // vertical treatment
    f3 = _mm_set1_epi32(filter_v[3]);
    f1 = _mm_set1_epi32(filter_v[1]);
    f2 = _mm_set1_epi32(filter_v[2]);
    f0 = _mm_set1_epi32(filter_v[0]);
    r0= _mm_setzero_si128();
    for (x = 0; x < width; x += 4) {
        x1 = _mm_loadl_epi64((__m128i *) &tmp[x]);
        x1 = _mm_cvtepi16_epi32(x1);
        x2 = _mm_loadl_epi64((__m128i *) &tmp[x + MAX_PB_SIZE]);
        x2 = _mm_cvtepi16_epi32(x2);
        x3 = _mm_loadl_epi64((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
        x3= _mm_cvtepi16_epi32(x3);
        for (y = 0; y < height; y++) {
            x0  = x1;
            x1  = x2;
            x2  = x3;
            x3  = _mm_loadl_epi64((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x3  = _mm_cvtepi16_epi32(x3);

            xx0 = _mm_mullo_epi32(x0,f0);
            xx1 = _mm_mullo_epi32(x1,f1);
            xx2 = _mm_mullo_epi32(x2,f2);
            xx3 = _mm_mullo_epi32(x3,f3);

            xx0 = _mm_add_epi32(xx0,xx1);
            xx2 = _mm_add_epi32(xx2,xx3);

            r1 = _mm_add_epi32(xx0,xx2);
            r1 = _mm_srai_epi32(r1,6);
            r1 = _mm_packs_epi32(r1,r0);

            _mm_storel_epi64((__m128i *) &dst[x], r1);
            tmp += MAX_PB_SIZE;
            dst += dststride;
        }
        dst = _dst;
        tmp = mcbuffer;
    }
}

void ff_hevc_put_hevc_epel_hv_8_2_sse(int16_t *dst, ptrdiff_t dststride,
                                    uint8_t *_src, ptrdiff_t _srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
                                    ptrdiff_t my) {

    int x, y;
    uint8_t *src = (uint8_t*) _src - 1;
    ptrdiff_t srcstride = _srcstride;
    const int8_t *filter_h = ff_hevc_epel_filters[mx - 1];
    const int8_t *filter_v = ff_hevc_epel_filters[my - 1];
    __m128i r0, bshuffle1, bshuffle2, x0, x1, x2, x3, f0, f1, f2, f3, r1;
    int16_t mcbuffer[(MAX_PB_SIZE + 3) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    r0 = _mm_set1_epi32(*(uint32_t*) filter_h);
    bshuffle1 = _mm_set_epi8(6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0);

    src -= EPEL_EXTRA_BEFORE * srcstride;

    bshuffle2=_mm_set_epi32(0,0,0,-1);

    for (y = 0; y < height + EPEL_EXTRA; y ++) {
        for(x = 0; x < width;x += 2){
            // load data in register
            x1 = _mm_loadl_epi64((__m128i *) &src[x]);
            x1 = _mm_shuffle_epi8(x1, bshuffle1);

            //  PMADDUBSW then PMADDW
            x1 = _mm_maddubs_epi16(x1, r0);
            x1 = _mm_hadd_epi16(x1, _mm_setzero_si128());

            // give results back
            *((uint32_t *)(tmp+x)) = _mm_cvtsi128_si32(x1);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = mcbuffer + EPEL_EXTRA_BEFORE * MAX_PB_SIZE;

    //vertical treatment
    f3 = _mm_set1_epi32(filter_v[3]);
    f1 = _mm_set1_epi32(filter_v[1]);
    f2 = _mm_set1_epi32(filter_v[2]);
    f0 = _mm_set1_epi32(filter_v[0]);
    r0= _mm_setzero_si128();
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 2) {
            x0 = _mm_loadl_epi64((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x1 = _mm_loadl_epi64((__m128i *) &tmp[x]);
            x2 = _mm_loadl_epi64((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x3 = _mm_loadl_epi64((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);

            x0= _mm_cvtepi16_epi32(x0);
            x1= _mm_cvtepi16_epi32(x1);
            x2= _mm_cvtepi16_epi32(x2);
            x3= _mm_cvtepi16_epi32(x3);

            x0= _mm_mullo_epi32(x0, f0);
            x1= _mm_mullo_epi32(x1, f1);
            x2= _mm_mullo_epi32(x2, f2);
            x3= _mm_mullo_epi32(x3, f3);

            x0= _mm_add_epi32(x0, x1);
            x2= _mm_add_epi32(x2, x3);

            r1= _mm_add_epi32(x0, x2);
            r1= _mm_srai_epi32(r1, 6);
            r1= _mm_packs_epi32(r1, r0);
            *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(r1);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}


void ff_hevc_put_hevc_epel_hv_10_sse(int16_t *dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t _srcstride, ptrdiff_t width, ptrdiff_t height, ptrdiff_t mx,
        ptrdiff_t my) {
    int x, y;
    uint16_t *src = (uint16_t*) _src;
    ptrdiff_t srcstride = _srcstride>>1;
    const int8_t *filter_h = ff_hevc_epel_filters[mx - 1];
    const int8_t *filter_v = ff_hevc_epel_filters[my - 1];
    __m128i r0, bshuffle1, bshuffle2, x0, x1, x2, x3, t0, t1, t2, t3, f0, f1,
    f2, f3, r1, r2, r3;
    int8_t filter_0 = filter_h[0];
    int8_t filter_1 = filter_h[1];
    int8_t filter_2 = filter_h[2];
    int8_t filter_3 = filter_h[3];
    int16_t mcbuffer[(MAX_PB_SIZE + 3) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;

    r0 = _mm_set_epi16(filter_3, filter_2, filter_1,
            filter_0, filter_3, filter_2, filter_1, filter_0);
    bshuffle1 = _mm_set_epi8(9,8,7,6,5,4, 3, 2,7,6,5,4, 3, 2, 1, 0);

    src -= EPEL_EXTRA_BEFORE * srcstride;

    f0 = _mm_set1_epi16(filter_v[0]);
    f1 = _mm_set1_epi16(filter_v[1]);
    f2 = _mm_set1_epi16(filter_v[2]);
    f3 = _mm_set1_epi16(filter_v[3]);


    /* horizontal treatment */
    if(!(width & 3)){
        bshuffle2 = _mm_set_epi8(13,12,11,10,9,8,7,6,11,10, 9,8,7,6,5, 4);
        for (y = 0; y < height + EPEL_EXTRA; y ++) {
            for(x=0;x<width;x+=4){

                x1 = _mm_loadu_si128((__m128i *) &src[x-1]);
                x2 = _mm_shuffle_epi8(x1, bshuffle1);
                x3 = _mm_shuffle_epi8(x1, bshuffle2);


                x2 = _mm_madd_epi16(x2, r0);
                x3 = _mm_madd_epi16(x3, r0);
                x2 = _mm_hadd_epi32(x2, x3);
                x2= _mm_srai_epi32(x2,2);   //>> (BIT_DEPTH - 8)

                x2 = _mm_packs_epi32(x2,r0);
                //give results back
                _mm_storel_epi64((__m128i *) &tmp[x], x2);

            }
            src += srcstride;
            tmp += MAX_PB_SIZE;
        }
        tmp = mcbuffer + EPEL_EXTRA_BEFORE * MAX_PB_SIZE;

        // vertical treatment


        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 4) {
                x0 = _mm_loadl_epi64((__m128i *) &tmp[x - MAX_PB_SIZE]);
                x1 = _mm_loadl_epi64((__m128i *) &tmp[x]);
                x2 = _mm_loadl_epi64((__m128i *) &tmp[x + MAX_PB_SIZE]);
                x3 = _mm_loadl_epi64((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);

                r0 = _mm_mullo_epi16(x0, f0);
                r1 = _mm_mulhi_epi16(x0, f0);
                r2 = _mm_mullo_epi16(x1, f1);
                t0 = _mm_unpacklo_epi16(r0, r1);

                r0 = _mm_mulhi_epi16(x1, f1);
                r1 = _mm_mullo_epi16(x2, f2);
                t1 = _mm_unpacklo_epi16(r2, r0);

                r2 = _mm_mulhi_epi16(x2, f2);
                r0 = _mm_mullo_epi16(x3, f3);
                t2 = _mm_unpacklo_epi16(r1, r2);

                r1 = _mm_mulhi_epi16(x3, f3);
                t3 = _mm_unpacklo_epi16(r0, r1);



                r0 = _mm_add_epi32(t0, t1);
                r0 = _mm_add_epi32(r0, t2);
                r0 = _mm_add_epi32(r0, t3);
                r0 = _mm_srai_epi32(r0, 6);

                // give results back
                r0 = _mm_packs_epi32(r0, r0);
                _mm_storel_epi64((__m128i *) &dst[x], r0);
            }
            tmp += MAX_PB_SIZE;
            dst += dststride;
        }
    }else{
        r1= _mm_setzero_si128();
        for (y = 0; y < height + EPEL_EXTRA; y ++) {
            for(x=0;x<width;x+=2){
                /* load data in register     */
                x1 = _mm_loadu_si128((__m128i *) &src[x-1]);
                x2 = _mm_shuffle_epi8(x1, bshuffle1);

                /*  PMADDUBSW then PMADDW     */
                x2 = _mm_madd_epi16(x2, r0);
                x2 = _mm_hadd_epi32(x2, r1);
                x2= _mm_srai_epi32(x2,2);   //>> (BIT_DEPTH - 8)
                x2 = _mm_packs_epi32(x2, r1);
                /* give results back            */
                *((uint32_t *)(tmp+x)) = _mm_cvtsi128_si32(x2);

            }
            src += srcstride;
            tmp += MAX_PB_SIZE;
        }

        tmp = mcbuffer + EPEL_EXTRA_BEFORE * MAX_PB_SIZE;

        /* vertical treatment */

        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 2) {
                /* check if memory needs to be reloaded */
                x0 = _mm_loadl_epi64((__m128i *) &tmp[x - MAX_PB_SIZE]);
                x1 = _mm_loadl_epi64((__m128i *) &tmp[x]);
                x2 = _mm_loadl_epi64((__m128i *) &tmp[x + MAX_PB_SIZE]);
                x3 = _mm_loadl_epi64((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);

                r0 = _mm_mullo_epi16(x0, f0);
                t0 = _mm_mulhi_epi16(x0, f0);

                x0= _mm_unpacklo_epi16(r0,t0);

                r1 = _mm_mullo_epi16(x1, f1);
                t1 = _mm_mulhi_epi16(x1, f1);

                x1= _mm_unpacklo_epi16(r1,t1);

                r2 = _mm_mullo_epi16(x2, f2);
                t2 = _mm_mulhi_epi16(x2, f2);

                x2= _mm_unpacklo_epi16(r2,t2);

                r3 = _mm_mullo_epi16(x3, f3);
                t3 = _mm_mulhi_epi16(x3, f3);

                x3= _mm_unpacklo_epi16(r3,t3);

                r0= _mm_add_epi32(x0,x1);
                r1= _mm_add_epi32(x2,x3);
                r0= _mm_add_epi32(r0,r1);
                r0 = _mm_srai_epi32(r0, 6);
                /* give results back            */
                r0 = _mm_packs_epi32(r0, r0);
                *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(r0);

            }
            tmp += MAX_PB_SIZE;
            dst += dststride;
        }
    }
}

void ff_hevc_put_hevc_qpel_pixels_10_sse(int16_t *dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    __m128i x1, x2, x4;
    uint16_t *src = (uint16_t*) _src;
    ptrdiff_t srcstride = _srcstride>>1;
    if(!(width & 7)){
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 8) {

                x1 = _mm_loadu_si128((__m128i *) &src[x]);
                x2 = _mm_slli_epi16(x1, 4); //14-BIT DEPTH
                _mm_store_si128((__m128i *) &dst[x], x2);

            }
            src += srcstride;
            dst += dststride;
        }
    }else if(!(width & 3)){
        for (y = 0; y < height; y++) {
            for(x=0;x<width;x+=4){
                x1 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_slli_epi16(x1, 4);//14-BIT DEPTH
                _mm_storel_epi64((__m128i *) &dst[x], x2);
            }
            src += srcstride;
            dst += dststride;
        }
    }else{
        x4= _mm_set_epi32(0,0,0,-1); //mask to store
        for (y = 0; y < height; y++) {
            for(x=0;x<width;x+=2){
                x1 = _mm_loadl_epi64((__m128i *) &src[x]);
                x2 = _mm_slli_epi16(x1, 4);//14-BIT DEPTH
                *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(x2);


            }
            src += srcstride;
            dst += dststride;
        }
    }


}

void ff_hevc_put_hevc_qpel_h8_1_v_1_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t* src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1, r2, r3, r4;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;
    int shift = 14 - 8;
    __m128i m0, m1,  m2,  m3,  m4,  m5,  m6,  m7;
    __m128i m8, m9, m10, m11, m12, m13, m14, m15;

    src -= ff_hevc_qpel_extra_before[1] * srcstride;
    m7  = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[0]);
    m8  = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[16]);
    m9  = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[32]);
    m10 = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[48]);

    for (y = 0; y < height + ff_hevc_qpel_extra[1]; y++) {
        for (x = 0; x < width; x += 8) {
            /* load data in register     */
            m0 = _mm_loadl_epi64((__m128i *) &src[x - 3]);
            m1 = _mm_loadl_epi64((__m128i *) &src[x - 2]);
            m2 = _mm_loadl_epi64((__m128i *) &src[x - 1]);
            m3 = _mm_loadl_epi64((__m128i *) &src[x]);
            m4 = _mm_loadl_epi64((__m128i *) &src[x + 1]);
            m5 = _mm_loadl_epi64((__m128i *) &src[x + 2]);

            m0 = _mm_unpacklo_epi8(m0, m1);
            m2 = _mm_unpacklo_epi8(m2, m3);
            m1 = _mm_loadl_epi64((__m128i *) &src[x + 3]);
            m3 = _mm_loadl_epi64((__m128i *) &src[x + 4]);

            m4 = _mm_unpacklo_epi8(m4, m5);
            m1 = _mm_unpacklo_epi8(m1, m3);

            /*  PMADDUBSW then PMADDW     */
            m0 = _mm_maddubs_epi16(m0, m7);
            m2 = _mm_maddubs_epi16(m2, m8);
            m4 = _mm_maddubs_epi16(m4, m9);
            m1 = _mm_maddubs_epi16(m1, m10);
            m0 = _mm_add_epi16(m0, m2);
            m4 = _mm_add_epi16(m4, m1);
            m0 = _mm_add_epi16(m0, m4);
            m0 = _mm_srai_epi32(m0, BIT_DEPTH - 8);

            /* give results back            */
            _mm_store_si128((__m128i *) &tmp[x], m0);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[1] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    const __m128i c1 = _mm_unpacklo_epi16(_mm_set1_epi16( -1), _mm_set1_epi16(  4));
    const __m128i c2 = _mm_unpacklo_epi16(_mm_set1_epi16(-10), _mm_set1_epi16( 58));
    const __m128i c3 = _mm_unpacklo_epi16(_mm_set1_epi16( 17), _mm_set1_epi16( -5));
    const __m128i c4 = _mm_unpacklo_epi16(_mm_set1_epi16(  1), _mm_setzero_si128());

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            t1 = _mm_unpacklo_epi16(x1, x2);
            x1 = _mm_unpackhi_epi16(x1, x2);
            t1 = _mm_madd_epi16(t1, c1);
            x1 = _mm_madd_epi16(x1, c1);

            t2 = _mm_unpacklo_epi16(x3, x4);
            x2 = _mm_unpackhi_epi16(x3, x4);
            t2 = _mm_madd_epi16(t2, c2);
            x2 = _mm_madd_epi16(x2, c2);

            t3 = _mm_unpacklo_epi16(x5, x6);
            x3 = _mm_unpackhi_epi16(x5, x6);
            t3 = _mm_madd_epi16(t3, c3);
            x3 = _mm_madd_epi16(x3, c3);

            t4 = _mm_unpacklo_epi16(x7, x8);
            x4 = _mm_unpackhi_epi16(x7, x8);
            t4 = _mm_madd_epi16(t4, c4);
            x4 = _mm_madd_epi16(x4, c4);


            t1 = _mm_add_epi32(t1, t2);
            t3 = _mm_add_epi32(t3, t4);
            t1 = _mm_add_epi32(t1, t3);
            t1 = _mm_srai_epi32(t1, 6);


            x1 = _mm_add_epi32(x1, x2);
            x3 = _mm_add_epi32(x3, x4);
            x1 = _mm_add_epi32(x1, x3);
            x1 = _mm_srai_epi32(x1, 6);

            t1 = _mm_packs_epi32(t1, x1);
            _mm_store_si128((__m128i *) &dst[x], t1);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h8_1_v_2_sse(int16_t *dst, ptrdiff_t dststride,
        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1, r2, r3, r4;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;
    int shift = 14 - 8;

    src -= ff_hevc_qpel_extra_before[2] * srcstride;
    r0 = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[0]);
    r1 = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[16]);
    r2 = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[32]);
    r3 = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[48]);

    for (y = 0; y < height + ff_hevc_qpel_extra[2]; y++) {
        for (x = 0; x < width; x += 8) {
            /* load data in register     */
            __m128i m0, m1, m2, m3, m4, m5, m6, m7;
            m0 = _mm_loadl_epi64((__m128i *) &src[x - 3]);
            m1 = _mm_loadl_epi64((__m128i *) &src[x - 2]);
            m2 = _mm_loadl_epi64((__m128i *) &src[x - 1]);
            m3 = _mm_loadl_epi64((__m128i *) &src[x]);
            m4 = _mm_loadl_epi64((__m128i *) &src[x + 1]);
            m5 = _mm_loadl_epi64((__m128i *) &src[x + 2]);
            m6 = _mm_loadl_epi64((__m128i *) &src[x + 3]);
            m7 = _mm_loadl_epi64((__m128i *) &src[x + 4]);

            x2 = _mm_unpacklo_epi8(m0, m1);
            x3 = _mm_unpacklo_epi8(m2, m3);
            x4 = _mm_unpacklo_epi8(m4, m5);
            x5 = _mm_unpacklo_epi8(m6, m7);

            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r1);
            x4 = _mm_maddubs_epi16(x4, r2);
            x5 = _mm_maddubs_epi16(x5, r3);
            x2 = _mm_add_epi16(x2, x3);
            x4 = _mm_add_epi16(x4, x5);
            x2 = _mm_add_epi16(x2, x4);
            x2 = _mm_srai_epi32(x2, BIT_DEPTH - 8);

            /* give results back            */
            _mm_store_si128((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = mcbuffer + ff_hevc_qpel_extra_before[2] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */

    const __m128i c0 = _mm_setzero_si128();
    const __m128i c1 = _mm_set1_epi16( -1);
    const __m128i c2 = _mm_set1_epi16(  4);
    const __m128i c3 = _mm_set1_epi16(-11);
    const __m128i c4 = _mm_set1_epi16( 40);
    const __m128i c5 = _mm_set1_epi16( 40);
    const __m128i c6 = _mm_set1_epi16(-11);
    const __m128i c7 = _mm_set1_epi16(  4);
    const __m128i c8 = _mm_set1_epi16( -1);
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            m1 = _mm_mullo_epi16(x1, c1);
            m2 = _mm_mulhi_epi16(x1, c1);
            t1 = _mm_unpacklo_epi16(m1, m2);
            x1 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x2, c2);
            m2 = _mm_mulhi_epi16(x2, c2);
            t2 = _mm_unpacklo_epi16(m1, m2);
            x2 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x3, c3);
            m2 = _mm_mulhi_epi16(x3, c3);
            t3 = _mm_unpacklo_epi16(m1, m2);
            x3 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x4, c4);
            m2 = _mm_mulhi_epi16(x4, c4);
            t4 = _mm_unpacklo_epi16(m1, m2);
            x4 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x5, c5);
            m2 = _mm_mulhi_epi16(x5, c5);
            t5 = _mm_unpacklo_epi16(m1, m2);
            x5 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x6, c6);
            m2 = _mm_mulhi_epi16(x6, c6);
            t6 = _mm_unpacklo_epi16(m1, m2);
            x6 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x7, c7);
            m2 = _mm_mulhi_epi16(x7, c7);
            t7 = _mm_unpacklo_epi16(m1, m2);
            x7 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x8, c8);
            m2 = _mm_mulhi_epi16(x8, c8);
            t8 = _mm_unpacklo_epi16(m1, m2);
            x8 = _mm_unpackhi_epi16(m1, m2);

            /* add calculus by correct value : */

            x1 = _mm_add_epi32(x1, x2); // x12
            x3 = _mm_add_epi32(x3, x4); // x34
            x5 = _mm_add_epi32(x5, x6); // x56
            x7 = _mm_add_epi32(x7, x8); // x78
            x1 = _mm_add_epi32(x1, x3); // x1234
            x5 = _mm_add_epi32(x5, x7); // x5678
            r1 = _mm_add_epi32(x1, x5); // x12345678
            r1 = _mm_srai_epi32(r1, 6);

            t1 = _mm_add_epi32(t1, t2); // t12
            t3 = _mm_add_epi32(t3, t4); // t34
            t5 = _mm_add_epi32(t5, t6); // t56
            t7 = _mm_add_epi32(t7, t8); // t78
            t1 = _mm_add_epi32(t1, t3); // t1234
            t5 = _mm_add_epi32(t5, t7); // t5678
            r0 = _mm_add_epi32(t1, t5); // t12345678
            r0 = _mm_srai_epi32(r0, 6);

            r0 = _mm_packs_epi32(r0, r1);
            _mm_store_si128((__m128i *) &dst[x], r0);
            
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}


void ff_hevc_put_hevc_qpel_h8_1_v_3_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1, r2, r3, r4;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;
    int shift = 14 - 8;

    src -= ff_hevc_qpel_extra_before[3] * srcstride;
    r0 = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[0]);
    r1 = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[16]);
    r2 = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[32]);
    r3 = _mm_load_si128((__m128i*) &qpel_hv_filter_1_8[48]);

    for (y = 0; y < height + ff_hevc_qpel_extra[3]; y++) {
        for (x = 0; x < width; x += 8) {
            /* load data in register     */
            __m128i m0, m1, m2, m3, m4, m5, m6, m7;
            m0 = _mm_loadl_epi64((__m128i *) &src[x - 3]);
            m1 = _mm_loadl_epi64((__m128i *) &src[x - 2]);
            m2 = _mm_loadl_epi64((__m128i *) &src[x - 1]);
            m3 = _mm_loadl_epi64((__m128i *) &src[x]);
            m4 = _mm_loadl_epi64((__m128i *) &src[x + 1]);
            m5 = _mm_loadl_epi64((__m128i *) &src[x + 2]);
            m6 = _mm_loadl_epi64((__m128i *) &src[x + 3]);
            m7 = _mm_loadl_epi64((__m128i *) &src[x + 4]);

            x2 = _mm_unpacklo_epi8(m0, m1);
            x3 = _mm_unpacklo_epi8(m2, m3);
            x4 = _mm_unpacklo_epi8(m4, m5);
            x5 = _mm_unpacklo_epi8(m6, m7);

            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r1);
            x4 = _mm_maddubs_epi16(x4, r2);
            x5 = _mm_maddubs_epi16(x5, r3);
            x2 = _mm_add_epi16(x2, x3);
            x4 = _mm_add_epi16(x4, x5);
            x2 = _mm_add_epi16(x2, x4);
            x2 = _mm_srai_epi32(x2, BIT_DEPTH - 8);

            /* give results back            */
            _mm_store_si128((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[3] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */
    const __m128i c0 = _mm_setzero_si128();
    const __m128i c1 = _mm_setzero_si128();
    const __m128i c2 = _mm_set1_epi16(  1);
    const __m128i c3 = _mm_set1_epi16( -5);
    const __m128i c4 = _mm_set1_epi16( 17);
    const __m128i c5 = _mm_set1_epi16( 58);
    const __m128i c6 = _mm_set1_epi16(-10);
    const __m128i c7 = _mm_set1_epi16(  4);
    const __m128i c8 = _mm_set1_epi16( -1);


    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            m1 = _mm_mullo_epi16(x1, c1);
            m2 = _mm_mulhi_epi16(x1, c1);
            t1 = _mm_unpacklo_epi16(m1, m2);
            x1 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x2, c2);
            m2 = _mm_mulhi_epi16(x2, c2);
            t2 = _mm_unpacklo_epi16(m1, m2);
            x2 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x3, c3);
            m2 = _mm_mulhi_epi16(x3, c3);
            t3 = _mm_unpacklo_epi16(m1, m2);
            x3 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x4, c4);
            m2 = _mm_mulhi_epi16(x4, c4);
            t4 = _mm_unpacklo_epi16(m1, m2);
            x4 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x5, c5);
            m2 = _mm_mulhi_epi16(x5, c5);
            t5 = _mm_unpacklo_epi16(m1, m2);
            x5 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x6, c6);
            m2 = _mm_mulhi_epi16(x6, c6);
            t6 = _mm_unpacklo_epi16(m1, m2);
            x6 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x7, c7);
            m2 = _mm_mulhi_epi16(x7, c7);
            t7 = _mm_unpacklo_epi16(m1, m2);
            x7 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x8, c8);
            m2 = _mm_mulhi_epi16(x8, c8);
            t8 = _mm_unpacklo_epi16(m1, m2);
            x8 = _mm_unpackhi_epi16(m1, m2);

            /* add calculus by correct value : */

            x1 = _mm_add_epi32(x1, x2); // x12
            x3 = _mm_add_epi32(x3, x4); // x34
            x5 = _mm_add_epi32(x5, x6); // x56
            x7 = _mm_add_epi32(x7, x8); // x78
            x1 = _mm_add_epi32(x1, x3); // x1234
            x5 = _mm_add_epi32(x5, x7); // x5678
            r1 = _mm_add_epi32(x1, x5); // x12345678
            r1 = _mm_srai_epi32(r1, 6);

            t1 = _mm_add_epi32(t1, t2); // t12
            t3 = _mm_add_epi32(t3, t4); // t34
            t5 = _mm_add_epi32(t5, t6); // t56
            t7 = _mm_add_epi32(t7, t8); // t78
            t1 = _mm_add_epi32(t1, t3); // t1234
            t5 = _mm_add_epi32(t5, t7); // t5678
            r0 = _mm_add_epi32(t1, t5); // t12345678
            r0 = _mm_srai_epi32(r0, 6);

            r0 = _mm_packs_epi32(r0, r1);
            _mm_store_si128((__m128i *) &dst[x], r0);
            
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}


void ff_hevc_put_hevc_qpel_h8_2_v_1_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[1] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -11, 40, 40, -11, 4, -1, -1, 4, -11, 40, 40, -11,
                      4, -1);

    for (y = 0; y < height + ff_hevc_qpel_extra[1]; y++) {
        for (x = 0; x < width; x += 8) {
            /* load data in register     */
            __m128i m0, m1, m2, m3, m4, m5, m6, m7;
            m0 = _mm_loadl_epi64((__m128i *) &src[x - 3]);
            m1 = _mm_loadl_epi64((__m128i *) &src[x - 2]);
            m2 = _mm_loadl_epi64((__m128i *) &src[x - 1]);
            m3 = _mm_loadl_epi64((__m128i *) &src[x]);
            m4 = _mm_loadl_epi64((__m128i *) &src[x + 1]);
            m5 = _mm_loadl_epi64((__m128i *) &src[x + 2]);
            m6 = _mm_loadl_epi64((__m128i *) &src[x + 3]);
            m7 = _mm_loadl_epi64((__m128i *) &src[x + 4]);
            x2 = _mm_unpacklo_epi64(m0, m1);
            x3 = _mm_unpacklo_epi64(m2, m3);
            x4 = _mm_unpacklo_epi64(m4, m5);
            x5 = _mm_unpacklo_epi64(m6, m7);

            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x4 = _mm_maddubs_epi16(x4, r0);
            x5 = _mm_maddubs_epi16(x5, r0);
            x2 = _mm_hadd_epi16(x2, x3);
            x4 = _mm_hadd_epi16(x4, x5);
            x2 = _mm_hadd_epi16(x2, x4);
            x2 = _mm_srli_si128(x2, BIT_DEPTH - 8);

            /* give results back            */
            _mm_store_si128((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[1] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */

    const __m128i c1 = _mm_unpacklo_epi16(_mm_set1_epi16( -1), _mm_set1_epi16(  4));
    const __m128i c2 = _mm_unpacklo_epi16(_mm_set1_epi16(-10), _mm_set1_epi16( 58));
    const __m128i c3 = _mm_unpacklo_epi16(_mm_set1_epi16( 17), _mm_set1_epi16( -5));
    const __m128i c4 = _mm_unpacklo_epi16(_mm_set1_epi16(  1), _mm_setzero_si128());

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            t1 = _mm_unpacklo_epi16(x1, x2);
            x1 = _mm_unpackhi_epi16(x1, x2);
            t1 = _mm_madd_epi16(t1, c1);
            x1 = _mm_madd_epi16(x1, c1);

            t2 = _mm_unpacklo_epi16(x3, x4);
            x2 = _mm_unpackhi_epi16(x3, x4);
            t2 = _mm_madd_epi16(t2, c2);
            x2 = _mm_madd_epi16(x2, c2);

            t3 = _mm_unpacklo_epi16(x5, x6);
            x3 = _mm_unpackhi_epi16(x5, x6);
            t3 = _mm_madd_epi16(t3, c3);
            x3 = _mm_madd_epi16(x3, c3);

            t4 = _mm_unpacklo_epi16(x7, x8);
            x4 = _mm_unpackhi_epi16(x7, x8);
            t4 = _mm_madd_epi16(t4, c4);
            x4 = _mm_madd_epi16(x4, c4);


            t1 = _mm_add_epi32(t1, t2);
            t3 = _mm_add_epi32(t3, t4);
            t1 = _mm_add_epi32(t1, t3);
            t1 = _mm_srai_epi32(t1, 6);


            x1 = _mm_add_epi32(x1, x2);
            x3 = _mm_add_epi32(x3, x4);
            x1 = _mm_add_epi32(x1, x3);
            x1 = _mm_srai_epi32(x1, 6);

            t1 = _mm_packs_epi32(t1, x1);
            _mm_store_si128((__m128i *) &dst[x], t1);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h8_2_v_2_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[2] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -11, 40, 40, -11, 4, -1,
                      -1, 4, -11, 40, 40, -11, 4, -1);

    for (y = 0; y < height + ff_hevc_qpel_extra[2]; y++) {
        for (x = 0; x < width; x += 8) {
            /* load data in register     */
            __m128i m0, m1, m2, m3, m4, m5, m6, m7;
            m0 = _mm_loadl_epi64((__m128i *) &src[x - 3]);
            m1 = _mm_loadl_epi64((__m128i *) &src[x - 2]);
            m2 = _mm_loadl_epi64((__m128i *) &src[x - 1]);
            m3 = _mm_loadl_epi64((__m128i *) &src[x]);
            m4 = _mm_loadl_epi64((__m128i *) &src[x + 1]);
            m5 = _mm_loadl_epi64((__m128i *) &src[x + 2]);
            m6 = _mm_loadl_epi64((__m128i *) &src[x + 3]);
            m7 = _mm_loadl_epi64((__m128i *) &src[x + 4]);
            x2 = _mm_unpacklo_epi64(m0, m1);
            x3 = _mm_unpacklo_epi64(m2, m3);
            x4 = _mm_unpacklo_epi64(m4, m5);
            x5 = _mm_unpacklo_epi64(m6, m7);

            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x4 = _mm_maddubs_epi16(x4, r0);
            x5 = _mm_maddubs_epi16(x5, r0);
            x2 = _mm_hadd_epi16(x2, x3);
            x4 = _mm_hadd_epi16(x4, x5);
            x2 = _mm_hadd_epi16(x2, x4);
            x2 = _mm_srli_si128(x2, BIT_DEPTH - 8);

            /* give results back            */
            _mm_store_si128((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[2] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */
    const __m128i c0 = _mm_setzero_si128();
    const __m128i c1 = _mm_set1_epi16( -1);
    const __m128i c2 = _mm_set1_epi16(  4);
    const __m128i c3 = _mm_set1_epi16(-11);
    const __m128i c4 = _mm_set1_epi16( 40);
    const __m128i c5 = _mm_set1_epi16( 40);
    const __m128i c6 = _mm_set1_epi16(-11);
    const __m128i c7 = _mm_set1_epi16(  4);
    const __m128i c8 = _mm_set1_epi16( -1);
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            m1 = _mm_mullo_epi16(x1, c1);
            m2 = _mm_mulhi_epi16(x1, c1);
            t1 = _mm_unpacklo_epi16(m1, m2);
            x1 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x2, c2);
            m2 = _mm_mulhi_epi16(x2, c2);
            t2 = _mm_unpacklo_epi16(m1, m2);
            x2 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x3, c3);
            m2 = _mm_mulhi_epi16(x3, c3);
            t3 = _mm_unpacklo_epi16(m1, m2);
            x3 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x4, c4);
            m2 = _mm_mulhi_epi16(x4, c4);
            t4 = _mm_unpacklo_epi16(m1, m2);
            x4 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x5, c5);
            m2 = _mm_mulhi_epi16(x5, c5);
            t5 = _mm_unpacklo_epi16(m1, m2);
            x5 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x6, c6);
            m2 = _mm_mulhi_epi16(x6, c6);
            t6 = _mm_unpacklo_epi16(m1, m2);
            x6 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x7, c7);
            m2 = _mm_mulhi_epi16(x7, c7);
            t7 = _mm_unpacklo_epi16(m1, m2);
            x7 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x8, c8);
            m2 = _mm_mulhi_epi16(x8, c8);
            t8 = _mm_unpacklo_epi16(m1, m2);
            x8 = _mm_unpackhi_epi16(m1, m2);

            /* add calculus by correct value : */

            x1 = _mm_add_epi32(x1, x2); // x12
            x3 = _mm_add_epi32(x3, x4); // x34
            x5 = _mm_add_epi32(x5, x6); // x56
            x7 = _mm_add_epi32(x7, x8); // x78
            x1 = _mm_add_epi32(x1, x3); // x1234
            x5 = _mm_add_epi32(x5, x7); // x5678
            r1 = _mm_add_epi32(x1, x5); // x12345678
            r1 = _mm_srai_epi32(r1, 6);

            t1 = _mm_add_epi32(t1, t2); // t12
            t3 = _mm_add_epi32(t3, t4); // t34
            t5 = _mm_add_epi32(t5, t6); // t56
            t7 = _mm_add_epi32(t7, t8); // t78
            t1 = _mm_add_epi32(t1, t3); // t1234
            t5 = _mm_add_epi32(t5, t7); // t5678
            r0 = _mm_add_epi32(t1, t5); // t12345678
            r0 = _mm_srai_epi32(r0, 6);

            r0 = _mm_packs_epi32(r0, r1);
            _mm_store_si128((__m128i *) &dst[x], r0);
            
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}


void ff_hevc_put_hevc_qpel_h8_2_v_3_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[3] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -11, 40, 40, -11, 4, -1, -1, 4, -11, 40, 40, -11,
                      4, -1);

    for (y = 0; y < height + ff_hevc_qpel_extra[3]; y++) {
        for (x = 0; x < width; x += 8) {
            /* load data in register     */
            __m128i m0, m1, m2, m3, m4, m5, m6, m7;
            m0 = _mm_loadl_epi64((__m128i *) &src[x - 3]);
            m1 = _mm_loadl_epi64((__m128i *) &src[x - 2]);
            m2 = _mm_loadl_epi64((__m128i *) &src[x - 1]);
            m3 = _mm_loadl_epi64((__m128i *) &src[x]);
            m4 = _mm_loadl_epi64((__m128i *) &src[x + 1]);
            m5 = _mm_loadl_epi64((__m128i *) &src[x + 2]);
            m6 = _mm_loadl_epi64((__m128i *) &src[x + 3]);
            m7 = _mm_loadl_epi64((__m128i *) &src[x + 4]);
            x2 = _mm_unpacklo_epi64(m0, m1);
            x3 = _mm_unpacklo_epi64(m2, m3);
            x4 = _mm_unpacklo_epi64(m4, m5);
            x5 = _mm_unpacklo_epi64(m6, m7);

            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x4 = _mm_maddubs_epi16(x4, r0);
            x5 = _mm_maddubs_epi16(x5, r0);
            x2 = _mm_hadd_epi16(x2, x3);
            x4 = _mm_hadd_epi16(x4, x5);
            x2 = _mm_hadd_epi16(x2, x4);
            x2 = _mm_srli_si128(x2, BIT_DEPTH - 8);

            /* give results back            */
            _mm_store_si128((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[3] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */
    const __m128i c0 = _mm_setzero_si128();
    const __m128i c1 = _mm_setzero_si128();
    const __m128i c2 = _mm_set1_epi16(  1);
    const __m128i c3 = _mm_set1_epi16( -5);
    const __m128i c4 = _mm_set1_epi16( 17);
    const __m128i c5 = _mm_set1_epi16( 58);
    const __m128i c6 = _mm_set1_epi16(-10);
    const __m128i c7 = _mm_set1_epi16(  4);
    const __m128i c8 = _mm_set1_epi16( -1);


    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            m1 = _mm_mullo_epi16(x1, c1);
            m2 = _mm_mulhi_epi16(x1, c1);
            t1 = _mm_unpacklo_epi16(m1, m2);
            x1 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x2, c2);
            m2 = _mm_mulhi_epi16(x2, c2);
            t2 = _mm_unpacklo_epi16(m1, m2);
            x2 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x3, c3);
            m2 = _mm_mulhi_epi16(x3, c3);
            t3 = _mm_unpacklo_epi16(m1, m2);
            x3 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x4, c4);
            m2 = _mm_mulhi_epi16(x4, c4);
            t4 = _mm_unpacklo_epi16(m1, m2);
            x4 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x5, c5);
            m2 = _mm_mulhi_epi16(x5, c5);
            t5 = _mm_unpacklo_epi16(m1, m2);
            x5 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x6, c6);
            m2 = _mm_mulhi_epi16(x6, c6);
            t6 = _mm_unpacklo_epi16(m1, m2);
            x6 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x7, c7);
            m2 = _mm_mulhi_epi16(x7, c7);
            t7 = _mm_unpacklo_epi16(m1, m2);
            x7 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x8, c8);
            m2 = _mm_mulhi_epi16(x8, c8);
            t8 = _mm_unpacklo_epi16(m1, m2);
            x8 = _mm_unpackhi_epi16(m1, m2);

            /* add calculus by correct value : */

            x1 = _mm_add_epi32(x1, x2); // x12
            x3 = _mm_add_epi32(x3, x4); // x34
            x5 = _mm_add_epi32(x5, x6); // x56
            x7 = _mm_add_epi32(x7, x8); // x78
            x1 = _mm_add_epi32(x1, x3); // x1234
            x5 = _mm_add_epi32(x5, x7); // x5678
            r1 = _mm_add_epi32(x1, x5); // x12345678
            r1 = _mm_srai_epi32(r1, 6);

            t1 = _mm_add_epi32(t1, t2); // t12
            t3 = _mm_add_epi32(t3, t4); // t34
            t5 = _mm_add_epi32(t5, t6); // t56
            t7 = _mm_add_epi32(t7, t8); // t78
            t1 = _mm_add_epi32(t1, t3); // t1234
            t5 = _mm_add_epi32(t5, t7); // t5678
            r0 = _mm_add_epi32(t1, t5); // t12345678
            r0 = _mm_srai_epi32(r0, 6);

            r0 = _mm_packs_epi32(r0, r1);
            _mm_store_si128((__m128i *) &dst[x], r0);
            
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}


void ff_hevc_put_hevc_qpel_h8_3_v_1_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[1] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -10, 58, 17, -5, 1, 0, 0, -1, 4, -10, 58, 17, -5, 1);


    for (y = 0; y < height + ff_hevc_qpel_extra[1]; y++) {
        for (x = 0; x < width; x +=8) {

            // load data in register
            x7 = _mm_loadl_epi64((__m128i *) &src[x-2]);
            x1 = _mm_unpacklo_epi64(x7, x7);
            x7 = _mm_loadl_epi64((__m128i *) &src[x]);
            x3= _mm_unpacklo_epi64(x7, x7);
            x7 = _mm_loadl_epi64((__m128i *) &src[x+2]);
            x4= _mm_unpacklo_epi64(x7, x7);
            x7 = _mm_loadl_epi64((__m128i *) &src[x+4]);
            x5= _mm_unpacklo_epi64(x7, x7);


            //  PMADDUBSW then PMADDW
            x1 = _mm_maddubs_epi16(x1, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x4 = _mm_maddubs_epi16(x4, r0);
            x5 = _mm_maddubs_epi16(x5, r0);
            x1 = _mm_hadd_epi16(x1, x3);
            x4 = _mm_hadd_epi16(x4, x5);
            x1 = _mm_hadd_epi16(x1, x4);
            x1 = _mm_srli_epi16(x1, BIT_DEPTH - 8);
            // give results back
            _mm_store_si128((__m128i*)(tmp+x),x1);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[1] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */
    const __m128i c1 = _mm_unpacklo_epi16(_mm_set1_epi16( -1), _mm_set1_epi16(  4));
    const __m128i c2 = _mm_unpacklo_epi16(_mm_set1_epi16(-10), _mm_set1_epi16( 58));
    const __m128i c3 = _mm_unpacklo_epi16(_mm_set1_epi16( 17), _mm_set1_epi16( -5));
    const __m128i c4 = _mm_unpacklo_epi16(_mm_set1_epi16(  1), _mm_setzero_si128());

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            t1 = _mm_unpacklo_epi16(x1, x2);
            x1 = _mm_unpackhi_epi16(x1, x2);
            t1 = _mm_madd_epi16(t1, c1);
            x1 = _mm_madd_epi16(x1, c1);

            t2 = _mm_unpacklo_epi16(x3, x4);
            x2 = _mm_unpackhi_epi16(x3, x4);
            t2 = _mm_madd_epi16(t2, c2);
            x2 = _mm_madd_epi16(x2, c2);

            t3 = _mm_unpacklo_epi16(x5, x6);
            x3 = _mm_unpackhi_epi16(x5, x6);
            t3 = _mm_madd_epi16(t3, c3);
            x3 = _mm_madd_epi16(x3, c3);

            t4 = _mm_unpacklo_epi16(x7, x8);
            x4 = _mm_unpackhi_epi16(x7, x8);
            t4 = _mm_madd_epi16(t4, c4);
            x4 = _mm_madd_epi16(x4, c4);


            t1 = _mm_add_epi32(t1, t2);
            t3 = _mm_add_epi32(t3, t4);
            t1 = _mm_add_epi32(t1, t3);
            t1 = _mm_srai_epi32(t1, 6);


            x1 = _mm_add_epi32(x1, x2);
            x3 = _mm_add_epi32(x3, x4);
            x1 = _mm_add_epi32(x1, x3);
            x1 = _mm_srai_epi32(x1, 6);

            t1 = _mm_packs_epi32(t1, x1);
            _mm_store_si128((__m128i *) &dst[x], t1);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h8_3_v_2_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[2] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -10, 58, 17, -5, 1, 0, 0, -1, 4, -10, 58, 17, -5, 1);


    for (y = 0; y < height + ff_hevc_qpel_extra[2]; y++) {
        for (x = 0; x < width; x +=8) {

            // load data in register
            x8 = _mm_loadl_epi64((__m128i *) &src[x-2]);
            x1 = _mm_unpacklo_epi64(x8, x8);
            x8 = _mm_loadl_epi64((__m128i *) &src[x]);
            x3= _mm_unpacklo_epi64(x8, x8);
            x8 = _mm_loadl_epi64((__m128i *) &src[x+2]);
            x4= _mm_unpacklo_epi64(x8, x8);
            x8 = _mm_loadl_epi64((__m128i *) &src[x+4]);
            x5= _mm_unpacklo_epi64(x8, x8);


            //  PMADDUBSW then PMADDW
            x1 = _mm_maddubs_epi16(x1, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x4 = _mm_maddubs_epi16(x4, r0);
            x5 = _mm_maddubs_epi16(x5, r0);
            x1 = _mm_hadd_epi16(x1, x3);
            x4 = _mm_hadd_epi16(x4, x5);
            x1 = _mm_hadd_epi16(x1, x4);
            x1 = _mm_srli_epi16(x1, BIT_DEPTH - 8);
            // give results back
            _mm_store_si128((__m128i*)(tmp+x),x1);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[2] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */

    const __m128i c0 = _mm_setzero_si128();
    const __m128i c1 = _mm_set1_epi16( -1);
    const __m128i c2 = _mm_set1_epi16(  4);
    const __m128i c3 = _mm_set1_epi16(-11);
    const __m128i c4 = _mm_set1_epi16( 40);
    const __m128i c5 = _mm_set1_epi16( 40);
    const __m128i c6 = _mm_set1_epi16(-11);
    const __m128i c7 = _mm_set1_epi16(  4);
    const __m128i c8 = _mm_set1_epi16( -1);
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            m1 = _mm_mullo_epi16(x1, c1);
            m2 = _mm_mulhi_epi16(x1, c1);
            t1 = _mm_unpacklo_epi16(m1, m2);
            x1 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x2, c2);
            m2 = _mm_mulhi_epi16(x2, c2);
            t2 = _mm_unpacklo_epi16(m1, m2);
            x2 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x3, c3);
            m2 = _mm_mulhi_epi16(x3, c3);
            t3 = _mm_unpacklo_epi16(m1, m2);
            x3 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x4, c4);
            m2 = _mm_mulhi_epi16(x4, c4);
            t4 = _mm_unpacklo_epi16(m1, m2);
            x4 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x5, c5);
            m2 = _mm_mulhi_epi16(x5, c5);
            t5 = _mm_unpacklo_epi16(m1, m2);
            x5 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x6, c6);
            m2 = _mm_mulhi_epi16(x6, c6);
            t6 = _mm_unpacklo_epi16(m1, m2);
            x6 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x7, c7);
            m2 = _mm_mulhi_epi16(x7, c7);
            t7 = _mm_unpacklo_epi16(m1, m2);
            x7 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x8, c8);
            m2 = _mm_mulhi_epi16(x8, c8);
            t8 = _mm_unpacklo_epi16(m1, m2);
            x8 = _mm_unpackhi_epi16(m1, m2);

            /* add calculus by correct value : */

            x1 = _mm_add_epi32(x1, x2); // x12
            x3 = _mm_add_epi32(x3, x4); // x34
            x5 = _mm_add_epi32(x5, x6); // x56
            x7 = _mm_add_epi32(x7, x8); // x78
            x1 = _mm_add_epi32(x1, x3); // x1234
            x5 = _mm_add_epi32(x5, x7); // x5678
            r1 = _mm_add_epi32(x1, x5); // x12345678
            r1 = _mm_srai_epi32(r1, 6);

            t1 = _mm_add_epi32(t1, t2); // t12
            t3 = _mm_add_epi32(t3, t4); // t34
            t5 = _mm_add_epi32(t5, t6); // t56
            t7 = _mm_add_epi32(t7, t8); // t78
            t1 = _mm_add_epi32(t1, t3); // t1234
            t5 = _mm_add_epi32(t5, t7); // t5678
            r0 = _mm_add_epi32(t1, t5); // t12345678
            r0 = _mm_srai_epi32(r0, 6);

            r0 = _mm_packs_epi32(r0, r1);
            _mm_store_si128((__m128i *) &dst[x], r0);
            
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h8_3_v_3_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[3] * srcstride;
    r0 = _mm_set_epi8(0,-1, 4, -10, 58, 17, -5, 1, 0, -1, 4, -10, 58, 17, -5, 1);

    r0 = _mm_set_epi8(-1, 4, -10, 58, 17, -5, 1, 0, 0, -1, 4, -10, 58, 17, -5, 1);
    x2= _mm_setzero_si128();
    for (y = 0; y < height + ff_hevc_qpel_extra[3]; y++) {
        for (x = 0; x < width; x +=8) {

            // load data in register
            x8 = _mm_loadl_epi64((__m128i *) &src[x-2]);
            x1 = _mm_unpacklo_epi64(x8, x8);
            x8 = _mm_loadl_epi64((__m128i *) &src[x]);
            x3= _mm_unpacklo_epi64(x8, x8);
            x8 = _mm_loadl_epi64((__m128i *) &src[x+2]);
            x4= _mm_unpacklo_epi64(x8, x8);
            x8 = _mm_loadl_epi64((__m128i *) &src[x+4]);
            x5= _mm_unpacklo_epi64(x8, x8);


            //  PMADDUBSW then PMADDW
            x1 = _mm_maddubs_epi16(x1, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x4 = _mm_maddubs_epi16(x4, r0);
            x5 = _mm_maddubs_epi16(x5, r0);
            x1 = _mm_hadd_epi16(x1, x3);
            x4 = _mm_hadd_epi16(x4, x5);
            x1 = _mm_hadd_epi16(x1, x4);
            x1 = _mm_srai_epi16(x1, BIT_DEPTH - 8);
            // give results back
            _mm_store_si128((__m128i*)(tmp+x),x1);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = mcbuffer + ff_hevc_qpel_extra_before[3] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    // vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers for register calculations

    const __m128i c0 = _mm_setzero_si128();
    const __m128i c1 = _mm_setzero_si128();
    const __m128i c2 = _mm_set1_epi16(  1);
    const __m128i c3 = _mm_set1_epi16( -5);
    const __m128i c4 = _mm_set1_epi16( 17);
    const __m128i c5 = _mm_set1_epi16( 58);
    const __m128i c6 = _mm_set1_epi16(-10);
    const __m128i c7 = _mm_set1_epi16(  4);
    const __m128i c8 = _mm_set1_epi16( -1);


    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            m1 = _mm_mullo_epi16(x1, c1);
            m2 = _mm_mulhi_epi16(x1, c1);
            t1 = _mm_unpacklo_epi16(m1, m2);
            x1 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x2, c2);
            m2 = _mm_mulhi_epi16(x2, c2);
            t2 = _mm_unpacklo_epi16(m1, m2);
            x2 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x3, c3);
            m2 = _mm_mulhi_epi16(x3, c3);
            t3 = _mm_unpacklo_epi16(m1, m2);
            x3 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x4, c4);
            m2 = _mm_mulhi_epi16(x4, c4);
            t4 = _mm_unpacklo_epi16(m1, m2);
            x4 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x5, c5);
            m2 = _mm_mulhi_epi16(x5, c5);
            t5 = _mm_unpacklo_epi16(m1, m2);
            x5 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x6, c6);
            m2 = _mm_mulhi_epi16(x6, c6);
            t6 = _mm_unpacklo_epi16(m1, m2);
            x6 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x7, c7);
            m2 = _mm_mulhi_epi16(x7, c7);
            t7 = _mm_unpacklo_epi16(m1, m2);
            x7 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x8, c8);
            m2 = _mm_mulhi_epi16(x8, c8);
            t8 = _mm_unpacklo_epi16(m1, m2);
            x8 = _mm_unpackhi_epi16(m1, m2);

            /* add calculus by correct value : */

            x1 = _mm_add_epi32(x1, x2); // x12
            x3 = _mm_add_epi32(x3, x4); // x34
            x5 = _mm_add_epi32(x5, x6); // x56
            x7 = _mm_add_epi32(x7, x8); // x78
            x1 = _mm_add_epi32(x1, x3); // x1234
            x5 = _mm_add_epi32(x5, x7); // x5678
            r1 = _mm_add_epi32(x1, x5); // x12345678
            r1 = _mm_srai_epi32(r1, 6);

            t1 = _mm_add_epi32(t1, t2); // t12
            t3 = _mm_add_epi32(t3, t4); // t34
            t5 = _mm_add_epi32(t5, t6); // t56
            t7 = _mm_add_epi32(t7, t8); // t78
            t1 = _mm_add_epi32(t1, t3); // t1234
            t5 = _mm_add_epi32(t5, t7); // t5678
            r0 = _mm_add_epi32(t1, t5); // t12345678
            r0 = _mm_srai_epi32(r0, 6);

            r0 = _mm_packs_epi32(r0, r1);
            _mm_store_si128((__m128i *) &dst[x], r0);
            
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

static av_always_inline void ff_hevc_put_qpel_hv4_h(int16_t *_dst, ptrdiff_t dststride,
                                                    uint8_t *_src, ptrdiff_t _srcstride,
                                                    int width, int height, __m128i r0,
                                                    int qpel_extra) {

    uint8_t *src = (uint8_t*) _src;
    int16_t *tmp = (int16_t*) _dst;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int x, y;

    for (y = 0; y < height + qpel_extra; y ++) {
        for (x = 0; x < width; x +=4) {
            __m128i x1, x3;
            const __m128i x2= _mm_setzero_si128();


            // load data in register
            x3 = _mm_loadl_epi64((__m128i *) &src[x-2]);
            x1 = _mm_unpacklo_epi64(x3, x3);
            x3 = _mm_loadl_epi64((__m128i *) &src[x]);
            x3= _mm_unpacklo_epi64(x3, x3);

            //  PMADDUBSW then PMADDW
            x1 = _mm_maddubs_epi16(x1, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x1 = _mm_hadd_epi16(x1, x3);
            x1 = _mm_hadd_epi16(x1, x2);
            x1 = _mm_srli_epi16(x1, BIT_DEPTH - 8);
            // give results back
            _mm_storel_epi64((__m128i*)(tmp+x),x1);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
}


void ff_hevc_put_hevc_qpel_h4_1_v_1_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t* src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, rBuffer, rTemp, r0, r1, r3;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;
    int shift = 14 - 8;
    const __m128i c0    = _mm_setzero_si128();

    src -= ff_hevc_qpel_extra_before[1] * srcstride;
    r0 = _mm_set_epi8(0, 1, -5, 17, 58, -10, 4, -1, 0, 1, -5, 17, 58, -10, 4,
                      -1);


    for (y = 0; y < height + ff_hevc_qpel_extra[1]; y ++) {
        for (x = 0; x < width; x += 4) {

            __m128i m0, m1, m2, m3, m4, m5, m6, m7;
            m0 = _mm_loadl_epi64((__m128i *) &src[x - 3]);
            m1 = _mm_loadl_epi64((__m128i *) &src[x - 2]);
            m2 = _mm_loadl_epi64((__m128i *) &src[x - 1]);
            m3 = _mm_loadl_epi64((__m128i *) &src[x]);
            x2 = _mm_unpacklo_epi64(m0, m1);
            x3 = _mm_unpacklo_epi64(m2, m3);


            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x2 = _mm_hadd_epi16(x2, x3);
            x2 = _mm_hadd_epi16(x2, _mm_setzero_si128());
            x2 = _mm_srli_epi16(x2, BIT_DEPTH - 8);
            _mm_storel_epi64((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[1] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */
    const __m128i c1 = _mm_set1_epi32( -1);
    const __m128i c2 = _mm_set1_epi32(  4);
    const __m128i c3 = _mm_set1_epi32(-10);
    const __m128i c4 = _mm_set1_epi32( 58);
    const __m128i c5 = _mm_set1_epi32( 17);
    const __m128i c6 = _mm_set1_epi32( -5);
    const __m128i c7 = _mm_set1_epi32(  1);
    const __m128i c8 = _mm_setzero_si128();
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 4) {
            x1 = _mm_loadu_si128((__m128i *) &tmp[x - 3 * srcstride]);
            x2 = _mm_loadu_si128((__m128i *) &tmp[x - 2 * srcstride]);
            x3 = _mm_loadu_si128((__m128i *) &tmp[x -     srcstride]);
            x4 = _mm_loadu_si128((__m128i *) &tmp[x                ]);
            x1 = _mm_cvtepi16_epi32(x1);
            x2 = _mm_cvtepi16_epi32(x2);
            x3 = _mm_cvtepi16_epi32(x3);
            x4 = _mm_cvtepi16_epi32(x4);
            r1 = _mm_mullo_epi32(x1, c1);
            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x2, c2));
            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x3, c3));
            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x4, c4));
            x1 = _mm_loadu_si128((__m128i *) &tmp[x +     srcstride]);
            x2 = _mm_loadu_si128((__m128i *) &tmp[x + 2 * srcstride]);
            x3 = _mm_loadu_si128((__m128i *) &tmp[x + 3 * srcstride]);
            x4 = _mm_loadu_si128((__m128i *) &tmp[x + 4 * srcstride]);
            x1 = _mm_cvtepi16_epi32(x1);
            x2 = _mm_cvtepi16_epi32(x2);
            x3 = _mm_cvtepi16_epi32(x3);
            x4 = _mm_cvtepi16_epi32(x4);
            r3 = _mm_mullo_epi32(x1, c5);
            r3 = _mm_add_epi32(r3, _mm_mullo_epi32(x2, c6));
            r3 = _mm_add_epi32(r3, _mm_mullo_epi32(x3, c7));
            r3 = _mm_add_epi32(r3, _mm_mullo_epi32(x4, c8));
            r1= _mm_add_epi32(r1,r3);
            r1 = _mm_srai_epi32 (r1, 14 - 8);
            r1 = _mm_packs_epi32(r1, c0);
            _mm_storel_epi64((__m128i *) &dst[x], r1);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h4_1_v_2_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[2] * srcstride;
    r0 = _mm_set_epi8(0, 1, -5, 17, 58, -10, 4, -1, 0, 1, -5, 17, 58, -10, 4,
                      -1);

    for (y = 0; y < height + ff_hevc_qpel_extra[2]; y++) {
        for (x = 0; x < width; x += 8) {
            /* load data in register     */
            __m128i m0, m1, m2, m3, m4, m5, m6, m7;
            m0 = _mm_loadu_si128((__m128i *) &src[x - 3]);
            m1 = _mm_loadu_si128((__m128i *) &src[x - 2]);
            m2 = _mm_loadu_si128((__m128i *) &src[x - 1]);
            m3 = _mm_loadu_si128((__m128i *) &src[x]);
            m4 = _mm_loadu_si128((__m128i *) &src[x + 1]);
            m5 = _mm_loadu_si128((__m128i *) &src[x + 2]);
            m6 = _mm_loadu_si128((__m128i *) &src[x + 3]);
            m7 = _mm_loadu_si128((__m128i *) &src[x + 4]);
            x2 = _mm_unpacklo_epi64(m0, m1);
            x3 = _mm_unpacklo_epi64(m2, m3);
            x4 = _mm_unpacklo_epi64(m4, m5);
            x5 = _mm_unpacklo_epi64(m6, m7);

            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x4 = _mm_maddubs_epi16(x4, r0);
            x5 = _mm_maddubs_epi16(x5, r0);
            x2 = _mm_hadd_epi16(x2, x3);
            x4 = _mm_hadd_epi16(x4, x5);
            x2 = _mm_hadd_epi16(x2, x4);
            x2 = _mm_srli_si128(x2, BIT_DEPTH - 8);

            /* give results back            */
            _mm_store_si128((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = mcbuffer + ff_hevc_qpel_extra_before[2] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */

    const __m128i c0 = _mm_setzero_si128();
    const __m128i c1 = _mm_set1_epi16( -1);
    const __m128i c2 = _mm_set1_epi16(  4);
    const __m128i c3 = _mm_set1_epi16(-11);
    const __m128i c4 = _mm_set1_epi16( 40);
    const __m128i c5 = _mm_set1_epi16( 40);
    const __m128i c6 = _mm_set1_epi16(-11);
    const __m128i c7 = _mm_set1_epi16(  4);
    const __m128i c8 = _mm_set1_epi16( -1);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            __m128i m1, m2;

            x1 = _mm_load_si128((__m128i *) &tmp[x - 3 * MAX_PB_SIZE]);
            x2 = _mm_load_si128((__m128i *) &tmp[x - 2 * MAX_PB_SIZE]);
            x3 = _mm_load_si128((__m128i *) &tmp[x - MAX_PB_SIZE]);
            x4 = _mm_load_si128((__m128i *) &tmp[x]);
            x5 = _mm_load_si128((__m128i *) &tmp[x + MAX_PB_SIZE]);
            x6 = _mm_load_si128((__m128i *) &tmp[x + 2 * MAX_PB_SIZE]);
            x7 = _mm_load_si128((__m128i *) &tmp[x + 3 * MAX_PB_SIZE]);
            x8 = _mm_load_si128((__m128i *) &tmp[x + 4 * MAX_PB_SIZE]);

            m1 = _mm_mullo_epi16(x1, c1);
            m2 = _mm_mulhi_epi16(x1, c1);
            t1 = _mm_unpacklo_epi16(m1, m2);
            x1 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x2, c2);
            m2 = _mm_mulhi_epi16(x2, c2);
            t2 = _mm_unpacklo_epi16(m1, m2);
            x2 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x3, c3);
            m2 = _mm_mulhi_epi16(x3, c3);
            t3 = _mm_unpacklo_epi16(m1, m2);
            x3 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x4, c4);
            m2 = _mm_mulhi_epi16(x4, c4);
            t4 = _mm_unpacklo_epi16(m1, m2);
            x4 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x5, c5);
            m2 = _mm_mulhi_epi16(x5, c5);
            t5 = _mm_unpacklo_epi16(m1, m2);
            x5 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x6, c6);
            m2 = _mm_mulhi_epi16(x6, c6);
            t6 = _mm_unpacklo_epi16(m1, m2);
            x6 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x7, c7);
            m2 = _mm_mulhi_epi16(x7, c7);
            t7 = _mm_unpacklo_epi16(m1, m2);
            x7 = _mm_unpackhi_epi16(m1, m2);

            m1 = _mm_mullo_epi16(x8, c8);
            m2 = _mm_mulhi_epi16(x8, c8);
            t8 = _mm_unpacklo_epi16(m1, m2);
            x8 = _mm_unpackhi_epi16(m1, m2);

            /* add calculus by correct value : */

            r1 = _mm_add_epi32(x1, x2); // x12
            x3 = _mm_add_epi32(x3, x4); // x34
            x5 = _mm_add_epi32(x5, x6); // x56
            x7 = _mm_add_epi32(x7, x8); // x78
            r1 = _mm_add_epi32(r1, x3); // x1234
            x7 = _mm_add_epi32(x5, x7); // x5678

            r0 = _mm_add_epi32(t1, t2); // t12
            t3 = _mm_add_epi32(t3, t4); // t34
            t5 = _mm_add_epi32(t5, t6); // t56
            t7 = _mm_add_epi32(t7, t8); // t78
            r0 = _mm_add_epi32(r0, t3); // t1234
            t7 = _mm_add_epi32(t5, t7); // t5678
            r1 = _mm_add_epi32(r1, x7); // x12345678
            r0 = _mm_add_epi32(r0, t7); // t12345678
            r1 = _mm_srai_epi32(r1, 6);
            r0 = _mm_srai_epi32(r0, 6);

            r0 = _mm_packs_epi32(r0, r1);
            _mm_store_si128((__m128i *) &dst[x], r0);

        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h4_1_v_3_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, rBuffer, rTemp, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[3] * srcstride;
    r0 = _mm_set_epi8(0, 1, -5, 17, 58, -10, 4, -1, 0, 1, -5, 17, 58, -10, 4,
                      -1);


    for (y = 0; y < height + ff_hevc_qpel_extra[3]; y ++) {
        for (x = 0; x < width; x += 4) {

            /* load data in register     */
            __m128i m0, m1, m2, m3;
            m0 = _mm_loadl_epi64((__m128i *) &src[x - 3]);
            m1 = _mm_loadl_epi64((__m128i *) &src[x - 2]);
            m2 = _mm_loadl_epi64((__m128i *) &src[x - 1]);
            m3 = _mm_loadl_epi64((__m128i *) &src[x]);
            x2 = _mm_unpacklo_epi64(m0, m1);
            x3 = _mm_unpacklo_epi64(m2, m3);

            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x2 = _mm_hadd_epi16(x2, x3);
            x2 = _mm_hadd_epi16(x2, _mm_setzero_si128());
            x2 = _mm_srli_epi16(x2, BIT_DEPTH - 8);
            /* give results back            */
            _mm_storel_epi64((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[3] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */
    t7 = _mm_set1_epi32(-1);
    t6 = _mm_set1_epi32(4);
    t5 = _mm_set1_epi32(-10);
    t4 = _mm_set1_epi32(58);
    t3 = _mm_set1_epi32(17);
    t2 = _mm_set1_epi32(-5);
    t1 = _mm_set1_epi32(1);
    t8= _mm_setzero_si128();


    for (y = 0; y < height; y ++) {
        for(x=0;x<width;x+=4){

            x1 = _mm_loadl_epi64((__m128i *) &tmp[x-2 * srcstride]);
            x2 = _mm_loadl_epi64((__m128i *) &tmp[x-srcstride]);
            x3 = _mm_loadl_epi64((__m128i *) &tmp[x]);
            x4 = _mm_loadl_epi64((__m128i *) &tmp[x+srcstride]);
            x5 = _mm_loadl_epi64((__m128i *) &tmp[x+2 * srcstride]);
            x6 = _mm_loadl_epi64((__m128i *) &tmp[x+3 * srcstride]);
            x7 = _mm_loadl_epi64((__m128i *) &tmp[x + 4 * srcstride]);

            x1 = _mm_cvtepi16_epi32(x1);
            x2 = _mm_cvtepi16_epi32(x2);
            x3 = _mm_cvtepi16_epi32(x3);
            x4 = _mm_cvtepi16_epi32(x4);
            x5 = _mm_cvtepi16_epi32(x5);
            x6 = _mm_cvtepi16_epi32(x6);
            x7 = _mm_cvtepi16_epi32(x7);


            r0 = _mm_mullo_epi32(x1, t1);

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x2,t2));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x3,t3));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x4,t4));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x5,t5));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x6,t6));
            
            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x7,t7));
            
            r0= _mm_srai_epi32(r0,6);
            
            r0= _mm_packs_epi32(r0,t8);
            
            
            _mm_storel_epi64((__m128i *) &dst[x], r0);
            
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h4_2_v_1_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[1] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -11, 40, 40, -11, 4, -1, -1, 4, -11, 40, 40, -11,
                      4, -1);


    for (y = 0; y < height + ff_hevc_qpel_extra[1]; y ++) {
        for (x = 0; x < width; x += 4) {

            /* load data in register     */
            __m128i m0, m1, m2, m3;
            m0 = _mm_loadu_si128((__m128i *) &src[x - 3]);
            m1 = _mm_loadu_si128((__m128i *) &src[x - 2]);
            m2 = _mm_loadu_si128((__m128i *) &src[x - 1]);
            m3 = _mm_loadu_si128((__m128i *) &src[x]);
            x2 = _mm_unpacklo_epi64(m0, m1);
            x3 = _mm_unpacklo_epi64(m2, m3);

            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x2 = _mm_hadd_epi16(x2, x3);
            x2 = _mm_hadd_epi16(x2, _mm_setzero_si128());
            x2 = _mm_srli_epi16(x2, BIT_DEPTH - 8);
            /* give results back            */
            _mm_storel_epi64((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[1] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */
    t7= _mm_set1_epi32(1);
    t6= _mm_set1_epi32(-5);
    t5= _mm_set1_epi32(17);
    t4= _mm_set1_epi32(58);
    t3= _mm_set1_epi32(-10);
    t2= _mm_set1_epi32(4);
    t1= _mm_set1_epi32(-1);
    t8= _mm_setzero_si128();

    for (y = 0; y < height; y ++) {
        for(x=0;x<width;x+=4){
            /* load data in register  */
            x1 = _mm_loadl_epi64((__m128i *) &tmp[x-(3 * srcstride)]);
            x2 = _mm_loadl_epi64((__m128i *) &tmp[x-(2 * srcstride)]);
            x3 = _mm_loadl_epi64((__m128i *) &tmp[x-srcstride]);
            x4 = _mm_loadl_epi64((__m128i *) &tmp[x]);
            x5 = _mm_loadl_epi64((__m128i *) &tmp[x+srcstride]);
            x6 = _mm_loadl_epi64((__m128i *) &tmp[x+(2 * srcstride)]);
            x7 = _mm_loadl_epi64((__m128i *) &tmp[x+(3 * srcstride)]);

            x1 = _mm_cvtepi16_epi32(x1);
            x2 = _mm_cvtepi16_epi32(x2);
            x3 = _mm_cvtepi16_epi32(x3);
            x4 = _mm_cvtepi16_epi32(x4);
            x5 = _mm_cvtepi16_epi32(x5);
            x6 = _mm_cvtepi16_epi32(x6);
            x7 = _mm_cvtepi16_epi32(x7);


            r1 = _mm_mullo_epi32(x1,t1);
            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x2,t2));
            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x3,t3));
            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x4,t4));
            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x5,t5));
            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x6,t6));
            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x7,t7));
            r1 = _mm_srai_epi32(r1,6);


            r1 = _mm_packs_epi32(r1,t8);

            // give results back
            _mm_storel_epi64((__m128i *) (dst + x), r1);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h4_2_v_2_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[2] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -11, 40, 40, -11, 4, -1, -1, 4, -11, 40, 40, -11,
                      4, -1);


    for (y = 0; y < height + ff_hevc_qpel_extra[2]; y ++) {
        for (x = 0; x < width; x += 4) {

            /* load data in register     */
            __m128i m0, m1, m2, m3, m4, m5, m6, m7;
            m0 = _mm_loadu_si128((__m128i *) &src[x - 3]);
            m1 = _mm_loadu_si128((__m128i *) &src[x - 2]);
            m2 = _mm_loadu_si128((__m128i *) &src[x - 1]);
            m3 = _mm_loadu_si128((__m128i *) &src[x]);
            x2 = _mm_unpacklo_epi64(m0, m1);
            x3 = _mm_unpacklo_epi64(m2, m3);


            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x2 = _mm_hadd_epi16(x2, x3);
            x2 = _mm_hadd_epi16(x2, _mm_setzero_si128());
            x2 = _mm_srli_epi16(x2, BIT_DEPTH - 8);
            /* give results back            */
            _mm_storel_epi64((__m128i *) &tmp[x], x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[2] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */

    t1= _mm_set1_epi32(-1);
    t2= _mm_set1_epi32(4);
    t3= _mm_set1_epi32(-11);
    t4= _mm_set1_epi32(40);
    t5= _mm_set1_epi32(40);
    t6= _mm_set1_epi32(-11);
    t7= _mm_set1_epi32(4);
    t8= _mm_set1_epi32(-1);


    x = 0;
    r0 = _mm_setzero_si128();
    for (y = 0; y < height; y ++) {
        for(x=0;x<width;x+=4){

            /* load data in register  */
            x1 = _mm_loadl_epi64((__m128i *) &tmp[x - 3 * srcstride]);
            x2 = _mm_loadl_epi64((__m128i *) &tmp[x-2 * srcstride]);
            x3 = _mm_loadl_epi64((__m128i *) &tmp[x-srcstride]);
            x4 = _mm_loadl_epi64((__m128i *) &tmp[x]);
            x5 = _mm_loadl_epi64((__m128i *) &tmp[x+srcstride]);
            x6 = _mm_loadl_epi64((__m128i *) &tmp[x+2 * srcstride]);
            x7 = _mm_loadl_epi64((__m128i *) &tmp[x+3 * srcstride]);
            x8 = _mm_loadl_epi64((__m128i *) &tmp[x + 4 * srcstride]);

            x1 = _mm_cvtepi16_epi32(x1);
            x2 = _mm_cvtepi16_epi32(x2);
            x3 = _mm_cvtepi16_epi32(x3);
            x4 = _mm_cvtepi16_epi32(x4);
            x5 = _mm_cvtepi16_epi32(x5);
            x6 = _mm_cvtepi16_epi32(x6);
            x7 = _mm_cvtepi16_epi32(x7);
            x8 = _mm_cvtepi16_epi32(x8);


            r1 = _mm_mullo_epi32(x1, t1);

            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x2,t2));


            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x3,t3));


            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x4,t4));


            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x5,t5));


            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x6,t6));


            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x7,t7));


            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x8,t8));


            r1= _mm_srai_epi32(r1,6);

            r1= _mm_packs_epi32(r1,t8);

            /* give results back            */
            _mm_storel_epi64((__m128i *) (dst+x), r1);

        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h4_2_v_3_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, rBuffer, r0;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[3] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -11, 40, 40, -11, 4, -1, -1, 4, -11, 40, 40, -11,
                      4, -1);

    rBuffer= _mm_set_epi32(0,0,0,-1);
    for (y = 0; y < height + ff_hevc_qpel_extra[3]; y ++) {
        for (x = 0; x < width; x += 2) {

            /* load data in register     */
            x1 = _mm_loadu_si128((__m128i *) &src[x-3]);
            x2 = _mm_unpacklo_epi64(x1, _mm_srli_si128(x1, 1));



            /*  PMADDUBSW then PMADDW     */
            x2 = _mm_maddubs_epi16(x2, r0);
            x2 = _mm_hadd_epi16(x2, r0);
            x2 = _mm_hadd_epi16(x2, _mm_setzero_si128());
            x2 = _mm_srli_epi16(x2, BIT_DEPTH - 8);
            /* give results back            */
            *((uint32_t *)(tmp+x)) = _mm_cvtsi128_si32(x2);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[3] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */

    t7 = _mm_set1_epi32(-1);
    t6 = _mm_set1_epi32(4);
    t5 = _mm_set1_epi32(-10);
    t4 = _mm_set1_epi32(58);
    t3 = _mm_set1_epi32(17);
    t2 = _mm_set1_epi32(-5);
    t1 = _mm_set1_epi32(1);
    t8= _mm_setzero_si128();


    for (y = 0; y < height; y ++) {
        for(x=0;x<width;x+=2){

            x1 = _mm_loadl_epi64((__m128i *) &tmp[x-2 * srcstride]);
            x2 = _mm_loadl_epi64((__m128i *) &tmp[x-srcstride]);
            x3 = _mm_loadl_epi64((__m128i *) &tmp[x]);
            x4 = _mm_loadl_epi64((__m128i *) &tmp[x+srcstride]);
            x5 = _mm_loadl_epi64((__m128i *) &tmp[x+2 * srcstride]);
            x6 = _mm_loadl_epi64((__m128i *) &tmp[x+3 * srcstride]);
            x7 = _mm_loadl_epi64((__m128i *) &tmp[x + 4 * srcstride]);

            x1 = _mm_cvtepi16_epi32(x1);
            x2 = _mm_cvtepi16_epi32(x2);
            x3 = _mm_cvtepi16_epi32(x3);
            x4 = _mm_cvtepi16_epi32(x4);
            x5 = _mm_cvtepi16_epi32(x5);
            x6 = _mm_cvtepi16_epi32(x6);
            x7 = _mm_cvtepi16_epi32(x7);


            r0 = _mm_mullo_epi32(x1, t1);

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x2,t2));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x3,t3));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x4,t4));
            
            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x5,t5));
            
            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x6,t6));
            
            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x7,t7));
            
            r0= _mm_srai_epi32(r0,6);
            
            r0= _mm_packs_epi32(r0,t8);
            
            *((uint32_t *)(dst+x)) = _mm_cvtsi128_si32(r0);
            
            
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h4_3_v_1_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[1] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -10, 58, 17, -5, 1, 0, 0, -1, 4, -10, 58, 17, -5, 1);
    x2= _mm_setzero_si128();

    for (y = 0; y < height + ff_hevc_qpel_extra[1]; y ++) {
        for (x = 0; x < width; x +=4) {

            // load data in register
            x3 = _mm_loadu_si128((__m128i *) &src[x-2]);
            x1 = _mm_unpacklo_epi64(x3, x3);
            x3 = _mm_srli_si128(x3,2);
            x3= _mm_unpacklo_epi64(x3, x3);

            //  PMADDUBSW then PMADDW
            x1 = _mm_maddubs_epi16(x1, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x1 = _mm_hadd_epi16(x1, x3);
            x1 = _mm_hadd_epi16(x1, x2);
            x1 = _mm_srli_epi16(x1, BIT_DEPTH - 8);
            // give results back
            _mm_storel_epi64((__m128i*)(tmp+x),x1);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[1] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */
    t7= _mm_set1_epi32(1);
    t6= _mm_set1_epi32(-5);
    t5= _mm_set1_epi32(17);
    t4= _mm_set1_epi32(58);
    t3= _mm_set1_epi32(-10);
    t2= _mm_set1_epi32(4);
    t1= _mm_set1_epi32(-1);
    t8= _mm_setzero_si128();

    for (y = 0; y < height; y ++) {
        for(x=0;x<width;x+=4){
            /* load data in register  */
            x1 = _mm_loadl_epi64((__m128i *) &tmp[x-(3 * srcstride)]);
            x2 = _mm_loadl_epi64((__m128i *) &tmp[x-(2 * srcstride)]);
            x3 = _mm_loadl_epi64((__m128i *) &tmp[x-srcstride]);
            x4 = _mm_loadl_epi64((__m128i *) &tmp[x]);
            x5 = _mm_loadl_epi64((__m128i *) &tmp[x+srcstride]);
            x6 = _mm_loadl_epi64((__m128i *) &tmp[x+(2 * srcstride)]);
            x7 = _mm_loadl_epi64((__m128i *) &tmp[x+(3 * srcstride)]);

            x1 = _mm_cvtepi16_epi32(x1);
            x2 = _mm_cvtepi16_epi32(x2);
            x3 = _mm_cvtepi16_epi32(x3);
            x4 = _mm_cvtepi16_epi32(x4);
            x5 = _mm_cvtepi16_epi32(x5);
            x6 = _mm_cvtepi16_epi32(x6);
            x7 = _mm_cvtepi16_epi32(x7);


            r1 = _mm_mullo_epi32(x1,t1);

            r1 = _mm_add_epi32(r1,
                               _mm_mullo_epi32(x2,t2));


            r1 = _mm_add_epi32(r1,
                               _mm_mullo_epi32(x3,t3));

            r1 = _mm_add_epi32(r1,
                               _mm_mullo_epi32(x4,t4));

            r1 = _mm_add_epi32(r1,
                               _mm_mullo_epi32(x5,t5));


            r1 = _mm_add_epi32(r1,
                               _mm_mullo_epi32(x6,t6));


            r1 = _mm_add_epi32(r1, _mm_mullo_epi32(x7,t7));
            r1 = _mm_srai_epi32(r1,6);


            r1 = _mm_packs_epi32(r1,t8);

            // give results back
            _mm_storel_epi64((__m128i *) (dst + x), r1);
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

void ff_hevc_put_hevc_qpel_h4_3_v_2_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, x8, r0, r1;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[2] * srcstride;
    r0 = _mm_set_epi8(-1, 4, -10, 58, 17, -5, 1, 0, 0, -1, 4, -10, 58, 17, -5, 1);
    x2= _mm_setzero_si128();

    for (y = 0; y < height + ff_hevc_qpel_extra[2]; y ++) {
        for (x = 0; x < width; x +=4) {

            // load data in register
            x3 = _mm_loadu_si128((__m128i *) &src[x-2]);
            x1 = _mm_unpacklo_epi64(x3, x3);
            x3 = _mm_srli_si128(x3,2);
            x3= _mm_unpacklo_epi64(x3, x3);

            //  PMADDUBSW then PMADDW
            x1 = _mm_maddubs_epi16(x1, r0);
            x3 = _mm_maddubs_epi16(x3, r0);
            x1 = _mm_hadd_epi16(x1, x3);
            x1 = _mm_hadd_epi16(x1, x2);
            x1 = _mm_srli_epi16(x1, BIT_DEPTH - 8);
            // give results back
            _mm_storel_epi64((__m128i*)(tmp+x),x1);

        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }
    tmp = mcbuffer + ff_hevc_qpel_extra_before[2] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    /* vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers
     for register calculations */
    t1= _mm_set1_epi32(-1);
    t2= _mm_set1_epi32(4);
    t3= _mm_set1_epi32(-11);
    t4= _mm_set1_epi32(40);
    t5= _mm_set1_epi32(40);
    t6= _mm_set1_epi32(-11);
    t7= _mm_set1_epi32(4);
    t8= _mm_set1_epi32(-1);

    {
        x = 0;
        r0 = _mm_setzero_si128();
        for (y = 0; y < height; y ++) {
            for(x=0;x<width;x+=4){

                /* load data in register  */
                x1 = _mm_loadl_epi64((__m128i *) &tmp[x - 3 * srcstride]);
                x2 = _mm_loadl_epi64((__m128i *) &tmp[x-2 * srcstride]);
                x3 = _mm_loadl_epi64((__m128i *) &tmp[x-srcstride]);
                x4 = _mm_loadl_epi64((__m128i *) &tmp[x]);
                x5 = _mm_loadl_epi64((__m128i *) &tmp[x+srcstride]);
                x6 = _mm_loadl_epi64((__m128i *) &tmp[x+2 * srcstride]);
                x7 = _mm_loadl_epi64((__m128i *) &tmp[x+3 * srcstride]);
                x8 = _mm_loadl_epi64((__m128i *) &tmp[x + 4 * srcstride]);

                x1 = _mm_cvtepi16_epi32(x1);
                x2 = _mm_cvtepi16_epi32(x2);
                x3 = _mm_cvtepi16_epi32(x3);
                x4 = _mm_cvtepi16_epi32(x4);
                x5 = _mm_cvtepi16_epi32(x5);
                x6 = _mm_cvtepi16_epi32(x6);
                x7 = _mm_cvtepi16_epi32(x7);
                x8 = _mm_cvtepi16_epi32(x8);


                r1 = _mm_mullo_epi32(x1, t1);

                r1 = _mm_add_epi32(r1,
                                   _mm_mullo_epi32(x2,t2));


                r1 = _mm_add_epi32(r1,
                                   _mm_mullo_epi32(x3,t3));


                r1 = _mm_add_epi32(r1,
                                   _mm_mullo_epi32(x4,t4));


                r1 = _mm_add_epi32(r1,
                                   _mm_mullo_epi32(x5,t5));


                r1 = _mm_add_epi32(r1,
                                   _mm_mullo_epi32(x6,t6));

                
                r1 = _mm_add_epi32(r1,
                                   _mm_mullo_epi32(x7,t7));
                
                
                r1 = _mm_add_epi32(r1,
                                   _mm_mullo_epi32(x8,t8));
                
                
                r1= _mm_srai_epi32(r1,6);
                
                r1= _mm_packs_epi32(r1,t8);
                
                /* give results back            */
                _mm_storel_epi64((__m128i *) (dst+x), r1);
                
            }
            tmp += MAX_PB_SIZE;
            dst += dststride;
        }
    }
}


void ff_hevc_put_hevc_qpel_h4_3_v_3_sse(int16_t *dst, ptrdiff_t dststride,
                                        uint8_t *_src, ptrdiff_t _srcstride, int width, int height) {
    int x, y;
    uint8_t *src = (uint8_t*) _src;
    ptrdiff_t srcstride = _srcstride / sizeof(uint8_t);
    int16_t mcbuffer[(MAX_PB_SIZE + 7) * MAX_PB_SIZE];
    int16_t *tmp = mcbuffer;
    __m128i x1, x2, x3, x4, x5, x6, x7, r0;
    __m128i t1, t2, t3, t4, t5, t6, t7, t8;

    src -= ff_hevc_qpel_extra_before[3] * srcstride;

    r0 = _mm_set_epi8(-1, 4, -10, 58, 17, -5, 1, 0, 0, -1, 4, -10, 58, 17, -5, 1);

    ff_hevc_put_qpel_hv4_h(tmp, dststride,
                           src, _srcstride, width, height, r0, ff_hevc_qpel_extra[3]);

    tmp = mcbuffer + ff_hevc_qpel_extra_before[3] * MAX_PB_SIZE;
    srcstride = MAX_PB_SIZE;

    // vertical treatment on temp table : tmp contains 16 bit values, so need to use 32 bit  integers for register calculations

    t7 = _mm_set1_epi32(-1);
    t6 = _mm_set1_epi32(4);
    t5 = _mm_set1_epi32(-10);
    t4 = _mm_set1_epi32(58);
    t3 = _mm_set1_epi32(17);
    t2 = _mm_set1_epi32(-5);
    t1 = _mm_set1_epi32(1);
    t8= _mm_setzero_si128();


    for (y = 0; y < height; y ++) {
        for(x=0;x<width;x+=4){

            x1 = _mm_loadl_epi64((__m128i *) &tmp[x-2 * srcstride]);
            x2 = _mm_loadl_epi64((__m128i *) &tmp[x-srcstride]);
            x3 = _mm_loadl_epi64((__m128i *) &tmp[x]);
            x4 = _mm_loadl_epi64((__m128i *) &tmp[x+srcstride]);
            x5 = _mm_loadl_epi64((__m128i *) &tmp[x + 2 * srcstride]);
            x6 = _mm_loadl_epi64((__m128i *) &tmp[x + 3 * srcstride]);
            x7 = _mm_loadl_epi64((__m128i *) &tmp[x + 4 * srcstride]);

            x1 = _mm_cvtepi16_epi32(x1);
            x2 = _mm_cvtepi16_epi32(x2);
            x3 = _mm_cvtepi16_epi32(x3);
            x4 = _mm_cvtepi16_epi32(x4);
            x5 = _mm_cvtepi16_epi32(x5);
            x6 = _mm_cvtepi16_epi32(x6);
            x7 = _mm_cvtepi16_epi32(x7);


            r0 = _mm_mullo_epi32(x1, t1);

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x2,t2));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x3,t3));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x4,t4));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x5,t5));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x6,t6));

            r0 = _mm_add_epi32(r0,
                               _mm_mullo_epi32(x7,t7));

            r0= _mm_srai_epi32(r0,6);

            r0= _mm_packs_epi32(r0,t8);
            
            
            _mm_storel_epi64((__m128i *) &dst[x], r0);
            
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}



