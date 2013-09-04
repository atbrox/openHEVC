/*
 * HEVC video Decoder
 *
 * Copyright (C) 2012 Guillaume Martres
 *
 * This file is part of Libav.
 *
 * Libav is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * Libav is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Libav; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/avassert.h"
#include "libavutil/pixdesc.h"
#include "get_bits.h"
#include "bit_depth_template.c"
#include "hevcdsp.h"
#include "hevc.h"
#include "up_sample_filter.h"
#define shift_1st 7
#define add_1st (1 << (shift_1st - 1))
#define shift_2nd (20 - BIT_DEPTH)
#define add_2nd (1 << (shift_2nd - 1))



#define SET(dst, x) (dst) = (x)
#define SCALE(dst, x) (dst) = av_clip_int16(((x) + add) >> shift)
#define ADD_AND_SCALE(dst, x) (dst) = av_clip_pixel((dst) + av_clip_int16(((x) + add) >> shift))



static void FUNC(copy_CTB)(uint8_t *_dst, uint8_t *_src, int width, int height, int _stride)
{
    pixel *dst = (pixel*)_dst;
    pixel *src = (pixel*)_src;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int i;

    for (i = 0; i < height; i++){
        memcpy(dst, src, width * sizeof(pixel));
        dst += stride;
        src += stride;
    }
}

static void FUNC(put_pcm)(uint8_t *_dst, ptrdiff_t _stride, int size,
                          GetBitContext *gb, int pcm_bit_depth)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);

    for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++)
            dst[x] = get_bits(gb, pcm_bit_depth) << (BIT_DEPTH - pcm_bit_depth);
        dst += stride;
    }
}
static void FUNC(dequant4x4)(int16_t *coeffs, int qp)
{
    int y;

    const uint8_t level_scale[] = { 40, 45, 51, 57, 64, 72 };

    //TODO: scaling_list_enabled_flag support

    int shift  = BIT_DEPTH - 3 - 4;
    int scale  = level_scale[qp % 6] << (qp / 6);
    int add    = 1 << (shift - 1);
    for (y = 0; y < 4*4; y+=4) {
        SCALE(coeffs[y+0], (coeffs[y+0] * scale));
        SCALE(coeffs[y+1], (coeffs[y+1] * scale));
        SCALE(coeffs[y+2], (coeffs[y+2] * scale));
        SCALE(coeffs[y+3], (coeffs[y+3] * scale));
    }

}
static void FUNC(dequant8x8)(int16_t *coeffs, int qp)
{
    int x, y;

    const uint8_t level_scale[] = { 40, 45, 51, 57, 64, 72 };

    //TODO: scaling_list_enabled_flag support

    int shift  = BIT_DEPTH - 2 - 4;
    int scale  = level_scale[qp % 6] << (qp / 6);
    int add    = 1 << (shift - 1);
    for (y = 0; y < 8*8; y+=8) {
        for (x = 0; x < 8; x++) {
            SCALE(coeffs[y+x], (coeffs[y+x] * scale));
        }
    }
}

static void FUNC(dequant16x16)(int16_t *coeffs, int qp)
{
    int x, y;

    const uint8_t level_scale[] = { 40, 45, 51, 57, 64, 72 };

    //TODO: scaling_list_enabled_flag support

    int shift  = BIT_DEPTH - 1 - 4;
    int scale  = level_scale[qp % 6] << (qp / 6);
    int add    = 1 << (shift - 1);
    for (y = 0; y < 16*16; y+=16) {
        for (x = 0; x < 16; x++) {
            SCALE(coeffs[y+x], (coeffs[y+x] * scale));
        }
    }
}

static void FUNC(dequant32x32)(int16_t *coeffs, int qp)
{
    int x, y;

    const uint8_t level_scale[] = { 40, 45, 51, 57, 64, 72 };

    //TODO: scaling_list_enabled_flag support

    int shift  = BIT_DEPTH - 4;
    int scale  = level_scale[qp % 6] << (qp / 6);
    int add    = 1 << (shift - 1);
    for (y = 0; y < 32*32; y+=32) {
        for (x = 0; x < 32; x++) {
            SCALE(coeffs[y+x], (coeffs[y+x] * scale));
        }
    }
}

static void FUNC(transquant_bypass4x4)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);

    for (y = 0; y < 4; y++) {
        for (x = 0; x < 4; x++) {
            dst[x] += *coeffs;
            coeffs++;
        }
        dst += stride;
    }

}

static void FUNC(transquant_bypass8x8)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);

    for (y = 0; y < 8; y++) {
        for (x = 0; x < 8; x++) {
            dst[x] += *coeffs;
            coeffs++;
        }
        dst += stride;
    }
}

static void FUNC(transquant_bypass16x16)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);

    for (y = 0; y < 16; y++) {
        for (x = 0; x < 16; x++) {
            dst[x] += *coeffs;
            coeffs++;
        }
        dst += stride;
    }

}

static void FUNC(transquant_bypass32x32)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);

    for (y = 0; y < 32; y++) {
        for (x = 0; x < 32; x++) {
            dst[x] += *coeffs;
            coeffs++;
        }
        dst += stride;
    }
}

static void FUNC(transform_skip)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int size = 4;
    int shift = 13 - BIT_DEPTH;
#if BIT_DEPTH <= 13
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif
    int x, y;
    switch (size){
        case 32:
            for (y = 0; y < 32*32; y+=32) {
                for (x = 0; x < 32; x++) {
                    dst[x] = av_clip_pixel(dst[x] + ((coeffs[y + x] + offset) >> shift));
                }
                dst += stride;
            }
            break;
        case 16:
            for (y = 0; y < 16*16; y+=16) {
                for (x = 0; x < 16; x++) {
                    dst[x] = av_clip_pixel(dst[x] + ((coeffs[y + x] + offset) >> shift));
                }
                dst += stride;
            }
            break;
        case 8:
            for (y = 0; y < 8*8; y+=8) {
                for (x = 0; x < 8; x++) {
                    dst[x] = av_clip_pixel(dst[x] + ((coeffs[y + x] + offset) >> shift));
                }
                dst += stride;
            }
            break;
        case 4:
            for (y = 0; y < 4*4; y+=4) {
                for (x = 0; x < 4; x++) {
                    dst[x] = av_clip_pixel(dst[x] + ((coeffs[y + x] + offset) >> shift));
                }
                dst += stride;
            }
            break;
    }
}

static void FUNC(transform_4x4_luma_add)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
#define TR_4x4_LUMA(dst, src, step, assign)                                     \
    do {                                                                        \
        int c0 = src[0*step] + src[2*step];                                     \
        int c1 = src[2*step] + src[3*step];                                     \
        int c2 = src[0*step] - src[3*step];                                     \
        int c3 = 74 * src[1*step];                                              \
                                                                                \
        assign(dst[2*step], 74 * (src[0*step] - src[2*step] + src[3*step]));    \
        assign(dst[0*step], 29 * c0 + 55 * c1 + c3);                            \
        assign(dst[1*step], 55 * c2 - 29 * c1 + c3);                            \
        assign(dst[3*step], 55 * c0 + 29 * c2 - c3);                            \
    } while (0)

    int i;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int shift = 7;
    int add = 1 << (shift - 1);
    int16_t *src = coeffs;

    for (i = 0; i < 4; i++) {
        TR_4x4_LUMA(src, src, 4, SCALE);
        src++;
    }

    shift = 20 - BIT_DEPTH;
    add = 1 << (shift - 1);
    for (i = 0; i < 4; i++) {
        TR_4x4_LUMA(dst, coeffs, 1, ADD_AND_SCALE);
        coeffs += 4;
        dst += stride;
    }

#undef TR_4x4_LUMA
}

#define TR_4(dst, src, dstep, sstep, assign)                                    \
    do {                                                                        \
        const int e0 = transform[8*0][0] * src[0*sstep] +                       \
                       transform[8*2][0] * src[2*sstep];                        \
        const int e1 = transform[8*0][1] * src[0*sstep] +                       \
                       transform[8*2][1] * src[2*sstep];                        \
        const int o0 = transform[8*1][0] * src[1*sstep] +                       \
                       transform[8*3][0] * src[3*sstep];                        \
        const int o1 = transform[8*1][1] * src[1*sstep] +                       \
                       transform[8*3][1] * src[3*sstep];                        \
                                                                                \
        assign(dst[0*dstep], e0 + o0);                                          \
        assign(dst[1*dstep], e1 + o1);                                          \
        assign(dst[2*dstep], e1 - o1);                                          \
        assign(dst[3*dstep], e0 - o0);                                          \
    } while (0)
#define TR_4_1(dst, src) TR_4(dst, src, 4, 4, SCALE)
#define TR_4_2(dst, src) TR_4(dst, src, 1, 1, ADD_AND_SCALE)

static void FUNC(transform_4x4_add)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
    int i;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int shift = 7;
    int add = 1 << (shift - 1);
    int16_t *src = coeffs;

    for (i = 0; i < 4; i++) {
        TR_4_1(src, src);
        src++;
    }

    shift = 20 - BIT_DEPTH;
    add = 1 << (shift - 1);
    for (i = 0; i < 4; i++) {
        TR_4_2(dst, coeffs);
        coeffs += 4;
        dst += stride;
    }
}

#define TR_8(dst, src, dstep, sstep, assign)                \
    do {                                                    \
        int i, j;                                           \
        int e_8[4];                                         \
        int o_8[4] = { 0 };                                 \
        for (i = 0; i < 4; i++)                             \
            for (j = 1; j < 8; j += 2)                      \
                o_8[i] += transform[4*j][i] * src[j*sstep]; \
        TR_4(e_8, src, 1, 2*sstep, SET);                    \
                                                            \
        for (i = 0; i < 4; i++) {                           \
            assign(dst[i*dstep], e_8[i] + o_8[i]);          \
            assign(dst[(7-i)*dstep], e_8[i] - o_8[i]);      \
        }                                                   \
    } while (0)
#define TR_16(dst, src, dstep, sstep, assign)                   \
    do {                                                        \
        int i, j;                                               \
        int e_16[8];                                            \
        int o_16[8] = { 0 };                                    \
        for (i = 0; i < 8; i++)                                 \
            for (j = 1; j < 16; j += 2)                         \
                o_16[i] += transform[2*j][i] * src[j*sstep];    \
        TR_8(e_16, src, 1, 2*sstep, SET);                       \
                                                                \
        for (i = 0; i < 8; i++) {                               \
            assign(dst[i*dstep], e_16[i] + o_16[i]);            \
            assign(dst[(15-i)*dstep], e_16[i] - o_16[i]);       \
        }                                                       \
    } while (0)
#define TR_32(dst, src, dstep, sstep, assign)               \
    do {                                                    \
        int i, j;                                           \
        int e_32[16];                                       \
        int o_32[16] = { 0 };                               \
        for (i = 0; i < 16; i++)                            \
            for (j = 1; j < 32; j += 2)                     \
                o_32[i] += transform[j][i] * src[j*sstep];  \
        TR_16(e_32, src, 1, 2*sstep, SET);                \
                                                            \
        for (i = 0; i < 16; i++) {                          \
            assign(dst[i*dstep], e_32[i] + o_32[i]);        \
            assign(dst[(31-i)*dstep], e_32[i] - o_32[i]);   \
        }                                                   \
    } while (0)

#define TR_8_1(dst, src) TR_8(dst, src, 8, 8, SCALE)
#define TR_16_1(dst, src) TR_16(dst, src, 16, 16, SCALE)
#define TR_32_1(dst, src) TR_32(dst, src, 32, 32, SCALE)

#define TR_8_2(dst, src) TR_8(dst, src, 1, 1, ADD_AND_SCALE)
#define TR_16_2(dst, src) TR_16(dst, src, 1, 1, ADD_AND_SCALE)
#define TR_32_2(dst, src) TR_32(dst, src, 1, 1, ADD_AND_SCALE)

static void FUNC(transform_8x8_add)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
    int i;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int shift = 7;
    int add = 1 << (shift - 1);
    int16_t *src = coeffs;

    for (i = 0; i < 8; i++) {
        TR_8_1(src, src);
        src++;
    }

    shift = 20 - BIT_DEPTH;
    add = 1 << (shift - 1);
    for (i = 0; i < 8; i++) {
        TR_8_2(dst, coeffs);
        coeffs += 8;
        dst += stride;
    }
}

static void FUNC(transform_16x16_add)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
    int i;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int shift = 7;
    int add = 1 << (shift - 1);
    int16_t *src = coeffs;

    for (i = 0; i < 16; i++) {
        TR_16_1(src, src);
        src++;
    }

    shift = 20 - BIT_DEPTH;
    add = 1 << (shift - 1);
    for (i = 0; i < 16; i++) {
        TR_16_2(dst, coeffs);
        coeffs += 16;
        dst += stride;
    }
}

static void FUNC(transform_32x32_add)(uint8_t *_dst, int16_t *coeffs, ptrdiff_t _stride)
{
#define IT32x32_even(i,w) ( src[ 0*w] * transform[ 0][i] ) + ( src[16*w] * transform[16][i] )
#define IT32x32_odd(i,w)  ( src[ 8*w] * transform[ 8][i] ) + ( src[24*w] * transform[24][i] )
#define IT16x16(i,w)      ( src[ 4*w] * transform[ 4][i] ) + ( src[12*w] * transform[12][i] ) + ( src[20*w] * transform[20][i] ) + ( src[28*w] * transform[28][i] )
#define IT8x8(i,w)        ( src[ 2*w] * transform[ 2][i] ) + ( src[ 6*w] * transform[ 6][i] ) + ( src[10*w] * transform[10][i] ) + ( src[14*w] * transform[14][i] ) + \
                          ( src[18*w] * transform[18][i] ) + ( src[22*w] * transform[22][i] ) + ( src[26*w] * transform[26][i] ) + ( src[30*w] * transform[30][i] )
#define IT4x4(i,w)        ( src[ 1*w] * transform[ 1][i] ) + ( src[ 3*w] * transform[ 3][i] ) + ( src[ 5*w] * transform[ 5][i] ) + ( src[ 7*w] * transform[ 7][i] ) + \
                          ( src[ 9*w] * transform[ 9][i] ) + ( src[11*w] * transform[11][i] ) + ( src[13*w] * transform[13][i] ) + ( src[15*w] * transform[15][i] ) + \
                          ( src[17*w] * transform[17][i] ) + ( src[19*w] * transform[19][i] ) + ( src[21*w] * transform[21][i] ) + ( src[23*w] * transform[23][i] ) + \
                          ( src[25*w] * transform[25][i] ) + ( src[27*w] * transform[27][i] ) + ( src[29*w] * transform[29][i] ) + ( src[31*w] * transform[31][i] )
    int i;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int shift = 7;
    int add = 1 << (shift - 1);
    int16_t *src = coeffs;

    for (i = 0; i < 32; i++) {
        TR_32_1(src, src);
        src++;
    }
    src   = coeffs;
    shift = 20 - BIT_DEPTH;
    add   = 1 << (shift - 1);
    for (i = 0; i < 32; i++) {
        TR_32_2(dst, coeffs);
        coeffs += 32;
        dst += stride;
    }
#undef IT32x32_even
#undef IT32x32_odd
#undef IT16x16
#undef IT8x8
#undef IT4x4
}

static void FUNC(sao_band_filter)(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride, SAOParams *sao,int *borders, int width, int height, int c_idx, int class)
{
    pixel *dst = (pixel*)_dst;
    pixel *src = (pixel*)_src;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int offset_table[32] = { 0 };
    int k, y, x;
    int chroma = c_idx!=0;
    int shift = BIT_DEPTH - 5;
    int *sao_offset_val = sao->offset_val[c_idx];
    int sao_left_class = sao->band_position[c_idx];
    int init_y = 0, init_x =0;

    switch (class) {
    case 0:
        if(!borders[2] )
            width -= ((8>>chroma)+2) ;
        if(!borders[3] )
            height -= ((4>>chroma)+2);
        break;
    case 1:
        init_y = -(4>>chroma)-2;
        if(!borders[2] )
            width -= ((8>>chroma)+2);
        height = (4>>chroma)+2;
        break;
    case 2:
        init_x = -(8>>chroma)-2;
        width = (8>>chroma)+2;
        if(!borders[3])
            height -= ((4>>chroma)+2);
        break;
    case 3:
        init_y = -(4>>chroma)-2;
        init_x = -(8>>chroma)-2;
        width = (8>>chroma)+2;
        height = (4>>chroma)+2;
        break;
    }

    dst = dst + (init_y*stride + init_x);
    src = src + (init_y*stride + init_x);
    for (k = 0; k < 4; k++)
        offset_table[(k + sao_left_class) & 31] = sao_offset_val[k + 1];
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            dst[x] = av_clip_pixel(src[x] + offset_table[src[x] >> shift]);
        }
        dst += stride;
        src += stride;
    }
}

static void FUNC(sao_band_filter_0)(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride, SAOParams *sao,int *borders, int width, int height, int c_idx) {
    FUNC(sao_band_filter)(_dst, _src, _stride, sao, borders, width, height, c_idx, 0);
}

static void FUNC(sao_band_filter_1)(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride, SAOParams *sao,int *borders, int width, int height, int c_idx) {
    FUNC(sao_band_filter)(_dst, _src, _stride, sao, borders, width, height, c_idx, 1);
}

static void FUNC(sao_band_filter_2)(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride, SAOParams *sao,int *borders, int width, int height, int c_idx) {
    FUNC(sao_band_filter)(_dst, _src, _stride, sao, borders, width, height, c_idx, 2);
}

static void FUNC(sao_band_filter_3)(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride, SAOParams *sao,int *borders, int width, int height, int c_idx) {
    FUNC(sao_band_filter)(_dst, _src, _stride, sao, borders, width, height, c_idx, 3);
}

static void FUNC(sao_edge_filter_0)(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride, SAOParams *sao,int *borders, int _width, int _height, int c_idx)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    pixel *src = (pixel*)_src;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int chroma = c_idx!=0;
    //struct SAOParams *sao;
    int *sao_offset_val = sao->offset_val[c_idx];
    int sao_eo_class = sao->eo_class[c_idx];

    const int8_t pos[4][2][2] = {
        { { -1,  0 }, {  1, 0 } }, // horizontal
        { {  0, -1 }, {  0, 1 } }, // vertical
        { { -1, -1 }, {  1, 1 } }, // 45 degree
        { {  1, -1 }, { -1, 1 } }, // 135 degree
    };
    const uint8_t edge_idx[] = { 1, 2, 0, 3, 4 };

    int init_x = 0, init_y = 0, width = _width, height = _height;

#define CMP(a, b) ((a) > (b) ? 1 : ((a) == (b) ? 0 : -1))

    if(!borders[2])
        width -= ((8>>chroma)+2) ;
    if(!borders[3])
        height -= ((4>>chroma)+2);

    dst = dst + (init_y*stride + init_x);
    src = src + (init_y*stride + init_x);
    init_y = init_x = 0;
    if (sao_eo_class != SAO_EO_VERT) {
        if (borders[0]) {
            int offset_val = sao_offset_val[0];
            int y_stride   = 0;
            for (y = 0; y < height; y++) {
                dst[y_stride] = av_clip_pixel(src[y_stride] + offset_val);
                y_stride += stride;
            }
            init_x = 1;
        }
        if (borders[2]) {
            int offset_val = sao_offset_val[0];
            int x_stride   = width-1;
            for (x = 0; x < height; x++) {
                dst[x_stride] = av_clip_pixel(src[x_stride] + offset_val);
                x_stride += stride;
            }
            width--;
        }

    }
    if (sao_eo_class != SAO_EO_HORIZ ) {
        if (borders[1]){
            int offset_val = sao_offset_val[0];
            for (x = init_x; x < width; x++) {
                dst[x] = av_clip_pixel(src[x] + offset_val);
            }
            init_y = 1;
        }
        if (borders[3]){
            int offset_val = sao_offset_val[0];
            int y_stride   = stride * (height-1);
            for (x = init_x; x < width; x++) {
                dst[x + y_stride] = av_clip_pixel(src[x + y_stride] + offset_val);
            }
            height--;
        }
    }
    {
        int y_stride     = init_y * stride;
        int pos_0_0      = pos[sao_eo_class][0][0];
        int pos_0_1      = pos[sao_eo_class][0][1];
        int pos_1_0      = pos[sao_eo_class][1][0];
        int pos_1_1      = pos[sao_eo_class][1][1];

        int y_stride_0_1 = (init_y + pos_0_1) * stride;
        int y_stride_1_1 = (init_y + pos_1_1) * stride;
        for (y = init_y; y < height; y++) {
            for (x = init_x; x < width; x++) {
                int diff0         = CMP(src[x + y_stride], src[x + pos_0_0 + y_stride_0_1]);
                int diff1         = CMP(src[x + y_stride], src[x + pos_1_0 + y_stride_1_1]);
                int offset_val    = edge_idx[2 + diff0 + diff1];
                dst[x + y_stride] = av_clip_pixel(src[x + y_stride] + sao_offset_val[offset_val]);
            }
            y_stride     += stride;
            y_stride_0_1 += stride;
            y_stride_1_1 += stride;
        }
    }
#undef CMP
}
static void FUNC(sao_edge_filter_1)(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride, SAOParams *sao,int *borders, int _width, int _height, int c_idx)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    pixel *src = (pixel*)_src;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int chroma = c_idx!=0;
    //struct SAOParams *sao;
    int *sao_offset_val = sao->offset_val[c_idx];
    int sao_eo_class = sao->eo_class[c_idx];

    const int8_t pos[4][2][2] = {
        { { -1,  0 }, {  1, 0 } }, // horizontal
        { {  0, -1 }, {  0, 1 } }, // vertical
        { { -1, -1 }, {  1, 1 } }, // 45 degree
        { {  1, -1 }, { -1, 1 } }, // 135 degree
    };
    const uint8_t edge_idx[] = { 1, 2, 0, 3, 4 };

    int init_x = 0, init_y = 0, width = _width, height = _height;

#define CMP(a, b) ((a) > (b) ? 1 : ((a) == (b) ? 0 : -1))

    init_y = -(4>>chroma)-2;
    if(!borders[2])
        width -= ((8>>chroma)+2);

    height = (4>>chroma)+2;

    dst = dst + (init_y*stride + init_x);
    src = src + (init_y*stride + init_x);
    init_y = init_x = 0;
    if (sao_eo_class != SAO_EO_VERT) {
        if (borders[0]) {
            int offset_val = sao_offset_val[0];
            int y_stride   = 0;
            for (y = 0; y < height; y++) {
                dst[y_stride] = av_clip_pixel(src[y_stride] + offset_val);
                y_stride += stride;
            }
            init_x = 1;
        }
        if (borders[2]) {
            int offset_val = sao_offset_val[0];
            int x_stride   = width-1;
            for (x = 0; x < height; x++) {
                dst[x_stride] = av_clip_pixel(src[x_stride] + offset_val);
                x_stride += stride;
            }
            width--;
        }

    }
    {
        int y_stride     = init_y * stride;
        int pos_0_0      = pos[sao_eo_class][0][0];
        int pos_0_1      = pos[sao_eo_class][0][1];
        int pos_1_0      = pos[sao_eo_class][1][0];
        int pos_1_1      = pos[sao_eo_class][1][1];

        int y_stride_0_1 = (init_y + pos_0_1) * stride;
        int y_stride_1_1 = (init_y + pos_1_1) * stride;
        for (y = init_y; y < height; y++) {
            for (x = init_x; x < width; x++) {
                int diff0         = CMP(src[x + y_stride], src[x + pos_0_0 + y_stride_0_1]);
                int diff1         = CMP(src[x + y_stride], src[x + pos_1_0 + y_stride_1_1]);
                int offset_val    = edge_idx[2 + diff0 + diff1];
                dst[x + y_stride] = av_clip_pixel(src[x + y_stride] + sao_offset_val[offset_val]);
            }
            y_stride     += stride;
            y_stride_0_1 += stride;
            y_stride_1_1 += stride;
        }
    }
#undef CMP
}
static void FUNC(sao_edge_filter_2)(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride, SAOParams *sao,int *borders, int _width, int _height, int c_idx)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    pixel *src = (pixel*)_src;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int chroma = c_idx!=0;
    //struct SAOParams *sao;
    int *sao_offset_val = sao->offset_val[c_idx];
    int sao_eo_class = sao->eo_class[c_idx];

    const int8_t pos[4][2][2] = {
        { { -1,  0 }, {  1, 0 } }, // horizontal
        { {  0, -1 }, {  0, 1 } }, // vertical
        { { -1, -1 }, {  1, 1 } }, // 45 degree
        { {  1, -1 }, { -1, 1 } }, // 135 degree
    };
    const uint8_t edge_idx[] = { 1, 2, 0, 3, 4 };

    int init_x = 0, init_y = 0, width = _width, height = _height;

#define CMP(a, b) ((a) > (b) ? 1 : ((a) == (b) ? 0 : -1))

    init_x = -(8>>chroma)-2;
    width = (8>>chroma)+2;
    if(!borders[3])
        height -= ((4>>chroma)+2);

    dst = dst + (init_y*stride + init_x);
    src = src + (init_y*stride + init_x);
    init_y = init_x = 0;
    if (sao_eo_class != SAO_EO_HORIZ) {
        if (borders[1]){
            int offset_val = sao_offset_val[0];
            for (x = init_x; x < width; x++) {
                dst[x] = av_clip_pixel(src[x] + offset_val);
            }
            init_y = 1;
        }
        if (borders[3]){
            int offset_val = sao_offset_val[0];
            int y_stride   = stride * (height-1);
            for (x = init_x; x < width; x++) {
                dst[x + y_stride] = av_clip_pixel(src[x + y_stride] + offset_val);
            }
            height--;
        }
    }
    {
        int y_stride     = init_y * stride;
        int pos_0_0      = pos[sao_eo_class][0][0];
        int pos_0_1      = pos[sao_eo_class][0][1];
        int pos_1_0      = pos[sao_eo_class][1][0];
        int pos_1_1      = pos[sao_eo_class][1][1];

        int y_stride_0_1 = (init_y + pos_0_1) * stride;
        int y_stride_1_1 = (init_y + pos_1_1) * stride;
        for (y = init_y; y < height; y++) {
            for (x = init_x; x < width; x++) {
                int diff0         = CMP(src[x + y_stride], src[x + pos_0_0 + y_stride_0_1]);
                int diff1         = CMP(src[x + y_stride], src[x + pos_1_0 + y_stride_1_1]);
                int offset_val    = edge_idx[2 + diff0 + diff1];
                dst[x + y_stride] = av_clip_pixel(src[x + y_stride] + sao_offset_val[offset_val]);
            }
            y_stride     += stride;
            y_stride_0_1 += stride;
            y_stride_1_1 += stride;
        }
    }
#undef CMP
}
static void FUNC(sao_edge_filter_3)(uint8_t *_dst, uint8_t *_src, ptrdiff_t _stride, SAOParams *sao,int *borders, int _width, int _height, int c_idx)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    pixel *src = (pixel*)_src;
    ptrdiff_t stride = _stride / sizeof(pixel);
    int chroma = c_idx!=0;
    //struct SAOParams *sao;
    int *sao_offset_val = sao->offset_val[c_idx];
    int sao_eo_class = sao->eo_class[c_idx];

    const int8_t pos[4][2][2] = {
        { { -1,  0 }, {  1, 0 } }, // horizontal
        { {  0, -1 }, {  0, 1 } }, // vertical
        { { -1, -1 }, {  1, 1 } }, // 45 degree
        { {  1, -1 }, { -1, 1 } }, // 135 degree
    };
    const uint8_t edge_idx[] = { 1, 2, 0, 3, 4 };

    int init_x = 0, init_y = 0, width = _width, height = _height;

#define CMP(a, b) ((a) > (b) ? 1 : ((a) == (b) ? 0 : -1))


    init_y = -(4>>chroma)-2;
    init_x = -(8>>chroma)-2;
    width = (8>>chroma)+2;
    height = (4>>chroma)+2;


    dst = dst + (init_y*stride + init_x);
    src = src + (init_y*stride + init_x);
    init_y = init_x = 0;

    {
        int y_stride     = init_y * stride;
        int pos_0_0      = pos[sao_eo_class][0][0];
        int pos_0_1      = pos[sao_eo_class][0][1];
        int pos_1_0      = pos[sao_eo_class][1][0];
        int pos_1_1      = pos[sao_eo_class][1][1];

        int y_stride_0_1 = (init_y + pos_0_1) * stride;
        int y_stride_1_1 = (init_y + pos_1_1) * stride;
        for (y = init_y; y < height; y++) {
            for (x = init_x; x < width; x++) {
                int diff0         = CMP(src[x + y_stride], src[x + pos_0_0 + y_stride_0_1]);
                int diff1         = CMP(src[x + y_stride], src[x + pos_1_0 + y_stride_1_1]);
                int offset_val    = edge_idx[2 + diff0 + diff1];
                dst[x + y_stride] = av_clip_pixel(src[x + y_stride] + sao_offset_val[offset_val]);
            }
            y_stride     += stride;
            y_stride_0_1 += stride;
            y_stride_1_1 += stride;
        }
    }
#undef CMP
}

#undef SET
#undef SCALE
#undef ADD_AND_SCALE
#undef TR_4
#undef TR_4_1
#undef TR_4_2
#undef TR_8
#undef TR_8_1
#undef TR_8_2
#undef TR_16
#undef TR_16_1
#undef TR_16_2
#undef TR_32
#undef TR_32_1
#undef TR_32_2

static void FUNC(put_hevc_qpel_pixels)(int16_t *dst, ptrdiff_t dststride,
                                       uint8_t *_src, ptrdiff_t _srcstride,
                                       int width, int height, int16_t* mcbuffer)
{
    int x, y;
    pixel *src = (pixel*)_src;
    ptrdiff_t srcstride = _srcstride/sizeof(pixel);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            dst[x] = src[x] << (14 - BIT_DEPTH);
        }
        src += srcstride;
        dst += dststride;
    }
}

#define QPEL_FILTER_1(src, stride)                                              \
    (-src[x-3*stride] + 4*src[x-2*stride] - 10*src[x-stride] + 58*src[x] +      \
     17*src[x+stride] - 5*src[x+2*stride] + 1*src[x+3*stride])
#define QPEL_FILTER_2(src, stride)                                              \
    (-src[x-3*stride] + 4*src[x-2*stride] - 11*src[x-stride] + 40*src[x] +      \
     40*src[x+stride] - 11*src[x+2*stride] + 4*src[x+3*stride] - src[x+4*stride])
#define QPEL_FILTER_3(src, stride)                                              \
    (src[x-2*stride] - 5*src[x-stride] + 17*src[x] + 58*src[x+stride]           \
     - 10*src[x+2*stride] + 4*src[x+3*stride] - src[x+4*stride])


#define PUT_HEVC_QPEL_H(H)                                                      \
static void FUNC(put_hevc_qpel_h ## H)(int16_t *dst, ptrdiff_t dststride,       \
                                          uint8_t *_src, ptrdiff_t _srcstride,  \
                                          int width, int height, int16_t* mcbuffer)                \
{                                                                               \
    int x, y;                                                                   \
    pixel *src = (pixel*)_src;                                                  \
    ptrdiff_t srcstride = _srcstride/sizeof(pixel);                             \
                                                                                \
    for (y = 0; y < height; y++) {                                              \
        for (x = 0; x < width; x++) {                                           \
            dst[x] = QPEL_FILTER_ ## H (src, 1) >> (BIT_DEPTH - 8);             \
        }                                                                       \
        src += srcstride;                                                       \
        dst += dststride;                                                       \
    }                                                                           \
}
#define PUT_HEVC_QPEL_V(V)                                                      \
static void FUNC(put_hevc_qpel_v ## V)(int16_t *dst, ptrdiff_t dststride,       \
                                          uint8_t *_src, ptrdiff_t _srcstride,  \
                                          int width, int height, int16_t* mcbuffer)                \
{                                                                               \
    int x, y;                                                                   \
    pixel *src = (pixel*)_src;                                                  \
    ptrdiff_t srcstride = _srcstride/sizeof(pixel);                             \
                                                                                \
    for (y = 0; y < height; y++)  {                                             \
        for (x = 0; x < width; x++) {                                           \
            dst[x] = QPEL_FILTER_ ## V (src, srcstride) >> (BIT_DEPTH - 8);     \
        }                                                                       \
        src += srcstride;                                                       \
        dst += dststride;                                                       \
    }                                                                           \
}
#define PUT_HEVC_QPEL_HV(H, V)                                                            \
static void FUNC(put_hevc_qpel_h ## H ## v ## V )(int16_t *dst, ptrdiff_t dststride,      \
                                                  uint8_t *_src, ptrdiff_t _srcstride,    \
                                                  int width, int height, int16_t* mcbuffer)                  \
{                                                                                         \
    int x, y;                                                                             \
    pixel *src = (pixel*)_src;                                                            \
    ptrdiff_t srcstride = _srcstride/sizeof(pixel);                                       \
                                                                                          \
    int16_t tmp_array[(MAX_PB_SIZE+7)*MAX_PB_SIZE];                                       \
    int16_t *tmp = tmp_array;                                                             \
                                                                                          \
    src -= ff_hevc_qpel_extra_before[V] * srcstride;                                      \
                                                                                          \
    for (y = 0; y < height + ff_hevc_qpel_extra[V]; y++) {                                \
        for (x = 0; x < width; x++) {                                                     \
            tmp[x] = QPEL_FILTER_ ## H (src, 1) >> (BIT_DEPTH - 8);                       \
        }                                                                                 \
        src += srcstride;                                                                 \
        tmp += MAX_PB_SIZE;                                                               \
    }                                                                                     \
                                                                                          \
    tmp = tmp_array + ff_hevc_qpel_extra_before[V] * MAX_PB_SIZE;                         \
                                                                                          \
    for (y = 0; y < height; y++) {                                                        \
        for (x = 0; x < width; x++) {                                                     \
            dst[x] = QPEL_FILTER_ ## V (tmp, MAX_PB_SIZE) >> 6;                           \
        }                                                                                 \
        tmp += MAX_PB_SIZE;                                                               \
        dst += dststride;                                                                 \
    }                                                                                     \
}

PUT_HEVC_QPEL_H(1)
PUT_HEVC_QPEL_H(2)
PUT_HEVC_QPEL_H(3)
PUT_HEVC_QPEL_V(1)
PUT_HEVC_QPEL_V(2)
PUT_HEVC_QPEL_V(3)
PUT_HEVC_QPEL_HV(1, 1)
PUT_HEVC_QPEL_HV(1, 2)
PUT_HEVC_QPEL_HV(1, 3)
PUT_HEVC_QPEL_HV(2, 1)
PUT_HEVC_QPEL_HV(2, 2)
PUT_HEVC_QPEL_HV(2, 3)
PUT_HEVC_QPEL_HV(3, 1)
PUT_HEVC_QPEL_HV(3, 2)
PUT_HEVC_QPEL_HV(3, 3)

static void FUNC(put_hevc_epel_pixels)(int16_t *dst, ptrdiff_t dststride,
                                       uint8_t *_src, ptrdiff_t _srcstride,
                                       int width, int height, int mx, int my,
                                       int16_t* mcbuffer)
{
    int x, y;
    pixel *src = (pixel*)_src;
    ptrdiff_t srcstride = _srcstride/sizeof(pixel);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            dst[x] = src[x] << (14 - BIT_DEPTH);
        }
        src += srcstride;
        dst += dststride;
    }
}

#define EPEL_FILTER(src, stride) \
    (filter_0*src[x-stride] + filter_1*src[x] + filter_2*src[x+stride] + filter_3*src[x+2*stride])

static void FUNC(put_hevc_epel_h)(int16_t *dst, ptrdiff_t dststride,
                                  uint8_t *_src, ptrdiff_t _srcstride,
                                  int width, int height, int mx, int my,
                                  int16_t* mcbuffer)
{
    int x, y;
    pixel *src = (pixel*)_src;
    ptrdiff_t srcstride = _srcstride/sizeof(pixel);
    const int8_t *filter = ff_hevc_epel_filters[mx-1];
    int8_t filter_0 = filter[0];
    int8_t filter_1 = filter[1];
    int8_t filter_2 = filter[2];
    int8_t filter_3 = filter[3];
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            dst[x] = EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_epel_v)(int16_t *dst, ptrdiff_t dststride,
                                  uint8_t *_src, ptrdiff_t _srcstride,
                                  int width, int height, int mx, int my,
                                  int16_t* mcbuffer)
{
    int x, y;
    pixel *src = (pixel*)_src;
    ptrdiff_t srcstride = _srcstride/sizeof(pixel);
    const int8_t *filter = ff_hevc_epel_filters[my-1];
    int8_t filter_0 = filter[0];
    int8_t filter_1 = filter[1];
    int8_t filter_2 = filter[2];
    int8_t filter_3 = filter[3];
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            dst[x] = EPEL_FILTER(src, srcstride) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        dst += dststride;
    }
}

static void FUNC(put_hevc_epel_hv)(int16_t *dst, ptrdiff_t dststride,
                                   uint8_t *_src, ptrdiff_t _srcstride,
                                   int width, int height, int mx, int my,int16_t* mcbuffer)
{
    int x, y;
    pixel *src = (pixel*)_src;
    ptrdiff_t srcstride = _srcstride/sizeof(pixel);
    const int8_t *filter_h = ff_hevc_epel_filters[mx-1];
    const int8_t *filter_v = ff_hevc_epel_filters[my-1];
    int8_t filter_0 = filter_h[0];
    int8_t filter_1 = filter_h[1];
    int8_t filter_2 = filter_h[2];
    int8_t filter_3 = filter_h[3];
    int16_t tmp_array[(MAX_PB_SIZE+3)*MAX_PB_SIZE];
    int16_t *tmp = tmp_array;

    src -= EPEL_EXTRA_BEFORE * srcstride;

    for (y = 0; y < height + EPEL_EXTRA; y++) {
        for (x = 0; x < width; x++) {
            tmp[x] = EPEL_FILTER(src, 1) >> (BIT_DEPTH - 8);
        }
        src += srcstride;
        tmp += MAX_PB_SIZE;
    }

    tmp = tmp_array + EPEL_EXTRA_BEFORE * MAX_PB_SIZE;
    filter_0 = filter_v[0];
    filter_1 = filter_v[1];
    filter_2 = filter_v[2];
    filter_3 = filter_v[3];
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            dst[x] = EPEL_FILTER(tmp, MAX_PB_SIZE) >> 6;
        }
        tmp += MAX_PB_SIZE;
        dst += dststride;
    }
}

static void FUNC(put_unweighted_pred)(uint8_t *_dst, ptrdiff_t _dststride,
                                      int16_t *src, ptrdiff_t srcstride,
                                      int width, int height)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t dststride = _dststride/sizeof(pixel);

    int shift = 14 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            dst[x] = av_clip_pixel((src[x] + offset) >> shift);
        }
        dst += dststride;
        src += srcstride;
    }
}

static void FUNC(put_weighted_pred_avg)(uint8_t *_dst, ptrdiff_t _dststride,
                                        int16_t *src1, int16_t *src2, ptrdiff_t srcstride,
                                        int width, int height)
{
    int x, y;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t dststride = _dststride/sizeof(pixel);

    int shift = 14 + 1 - BIT_DEPTH;
#if BIT_DEPTH < 14
    int offset = 1 << (shift - 1);
#else
    int offset = 0;
#endif
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            dst[x] = av_clip_pixel((src1[x] + src2[x] + offset) >> shift);
        }
        dst  += dststride;
        src1 += srcstride;
        src2 += srcstride;
    }
}

static void FUNC(weighted_pred)(uint8_t denom, int16_t wlxFlag, int16_t olxFlag,
                                     uint8_t *_dst, ptrdiff_t _dststride,
                                     int16_t *src, ptrdiff_t srcstride,
                                     int width, int height)
{
    int shift;
    int log2Wd;
    int wx;
    int ox;
    int x , y;
    int offset;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t dststride = _dststride/sizeof(pixel);

    shift = 14 - BIT_DEPTH;
    log2Wd = denom + shift;
    offset = 1 << (log2Wd - 1);
    wx = wlxFlag;
    ox = olxFlag * ( 1 << ( BIT_DEPTH - 8 ) );

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (log2Wd >= 1) {
                dst[x] = av_clip_pixel(((src[x] * wx + offset) >> log2Wd) + ox);
            } else {
                dst[x] = av_clip_pixel(src[x] * wx + ox);
            }
        }
        dst  += dststride;
        src  += srcstride;
    }
}

static void FUNC(weighted_pred_avg)(uint8_t denom, int16_t wl0Flag, int16_t wl1Flag,
                                         int16_t ol0Flag, int16_t ol1Flag, uint8_t *_dst, ptrdiff_t _dststride,
                                         int16_t *src1, int16_t *src2, ptrdiff_t srcstride,
                                         int width, int height)
{
    int shift;
    int log2Wd;
    int w0;
    int w1;
    int o0;
    int o1;
    int x , y;
    pixel *dst = (pixel*)_dst;
    ptrdiff_t dststride = _dststride/sizeof(pixel);

    shift = 14 - BIT_DEPTH;
    log2Wd = denom + shift;
    w0 = wl0Flag;
    w1 = wl1Flag;
    o0 = (ol0Flag) * ( 1 << ( BIT_DEPTH - 8 ) );
    o1 = (ol1Flag) * ( 1 << ( BIT_DEPTH - 8 ) );

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            dst[x] = av_clip_pixel((src1[x] * w0 + src2[x] * w1 + ((o0 + o1 + 1) << log2Wd)) >> (log2Wd + 1));
        }
        dst  += dststride;
        src1 += srcstride;
        src2 += srcstride;
    }
}


// line zero
#define P3 pix[-4*xstride]
#define P2 pix[-3*xstride]
#define P1 pix[-2*xstride]
#define P0 pix[-xstride]
#define Q0 pix[0]
#define Q1 pix[xstride]
#define Q2 pix[2*xstride]
#define Q3 pix[3*xstride]

// line three. used only for deblocking decision
#define TP3 pix[-4*xstride+3*ystride]
#define TP2 pix[-3*xstride+3*ystride]
#define TP1 pix[-2*xstride+3*ystride]
#define TP0 pix[-xstride+3*ystride]
#define TQ0 pix[3*ystride]
#define TQ1 pix[xstride+3*ystride]
#define TQ2 pix[2*xstride+3*ystride]
#define TQ3 pix[3*xstride+3*ystride]

static void FUNC(hevc_loop_filter_luma)(uint8_t *_pix, ptrdiff_t _xstride, ptrdiff_t _ystride, int *_beta, int *_tc,
                                        uint8_t *_no_p, uint8_t *_no_q)
{
    int d, j;
    pixel *pix = (pixel*)_pix;
    ptrdiff_t xstride = _xstride/sizeof(pixel);
    ptrdiff_t ystride = _ystride/sizeof(pixel);

    for (j = 0; j < 2; j++) {
        const int dp0 = abs(P2 - 2 * P1 +  P0);
        const int dq0 = abs(Q2 - 2 * Q1 +  Q0);
        const int dp3 = abs(TP2 - 2 * TP1 + TP0);
        const int dq3 = abs(TQ2 - 2 * TQ1 + TQ0);
        const int d0 = dp0 + dq0;
        const int d3 = dp3 + dq3;
        int beta = _beta[j] << (BIT_DEPTH - 8);
        const int tc = _tc[j] << (BIT_DEPTH - 8);
        const int no_p = _no_p[j];
        const int no_q = _no_q[j];

        if (d0 + d3 >= beta /*|| tc <= 0*/) {
            pix += 4*ystride;
            continue;
        } else {
            const int beta_3 = beta >> 3;
            const int beta_2 = beta >> 2;
            const int tc25 = ((tc * 5 + 1) >> 1);

            if(abs( P3 -  P0) + abs( Q3 -  Q0) < beta_3 && abs( P0 -  Q0) < tc25 &&
               abs(TP3 - TP0) + abs(TQ3 - TQ0) < beta_3 && abs(TP0 - TQ0) < tc25 &&
                                     (d0 << 1) < beta_2 &&      (d3 << 1) < beta_2) {
                // strong filtering
                const int tc2 = tc << 1;
                for(d = 0; d < 4; d++) {
                    const int p3 = P3;
                    const int p2 = P2;
                    const int p1 = P1;
                    const int p0 = P0;
                    const int q0 = Q0;
                    const int q1 = Q1;
                    const int q2 = Q2;
                    const int q3 = Q3;
                    if(!no_p) {
                        P0 = p0 + av_clip((( p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4 ) >> 3) - p0, -tc2, tc2);
                        P1 = p1 + av_clip((( p2 + p1 + p0 + q0 + 2 ) >> 2) - p1, -tc2, tc2);
                        P2 = p2 + av_clip((( 2*p3 + 3*p2 + p1 + p0 + q0 + 4 ) >> 3) - p2, -tc2, tc2);
                    }
                    if(!no_q) {
                        Q0 = q0 + av_clip((( p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4 ) >> 3) - q0, -tc2, tc2);
                        Q1 = q1 + av_clip((( p0 + q0 + q1 + q2 + 2 ) >> 2) - q1, -tc2, tc2);
                        Q2 = q2 + av_clip((( 2*q3 + 3*q2 + q1 + q0 + p0 + 4 ) >> 3) - q2, -tc2, tc2);
                    }
                    pix += ystride;
                }
            } else { // normal filtering
                int nd_p = 1;
                int nd_q = 1;
                const int tc_2 = tc >> 1;
                if (dp0 + dp3 < ((beta+(beta>>1))>>3))
                    nd_p = 2;
                if (dq0 + dq3 < ((beta+(beta>>1))>>3))
                    nd_q = 2;

                for(d = 0; d < 4; d++) {
                    const int p2 = P2;
                    const int p1 = P1;
                    const int p0 = P0;
                    const int q0 = Q0;
                    const int q1 = Q1;
                    const int q2 = Q2;
                    int delta0 = (9*(q0 - p0) - 3*(q1 - p1) + 8) >> 4;
                    if (abs(delta0) < 10 * tc) {
                        delta0 = av_clip(delta0, -tc, tc);
                        if(!no_p)
                            P0 = av_clip_pixel(p0 + delta0);
                        if(!no_q)
                            Q0 = av_clip_pixel(q0 - delta0);
                        if(!no_p && nd_p > 1) {
                            const int deltap1 = av_clip((((p2 + p0 + 1) >> 1) - p1 + delta0) >> 1, -tc_2, tc_2);
                            P1 = av_clip_pixel(p1 + deltap1);
                        }
                        if(!no_q && nd_q > 1) {
                            const int deltaq1 = av_clip((((q2 + q0 + 1) >> 1) - q1 - delta0) >> 1, -tc_2, tc_2);
                            Q1 = av_clip_pixel(q1 + deltaq1);
                        }
                    }
                    pix += ystride;
                }
            }
        }
    }
}


static void FUNC(hevc_loop_filter_chroma)(uint8_t *_pix, ptrdiff_t _xstride, ptrdiff_t _ystride, int *_tc, uint8_t *_no_p, uint8_t *_no_q)
{
    int d, j;
    int no_p, no_q;
    pixel *pix = (pixel*)_pix;
    ptrdiff_t xstride = _xstride/sizeof(pixel);
    ptrdiff_t ystride = _ystride/sizeof(pixel);

    for(j = 0; j < 2; j++) {
        const int tc = _tc[j] << (BIT_DEPTH - 8);
        if( tc <= 0 ) {
            pix += 4 * ystride;
            continue;
        }
        no_p = _no_p[j];
        no_q = _no_q[j];

        for(d = 0; d < 4; d++) {
            int delta0;
            const int p1 = P1;
            const int p0 = P0;
            const int q0 = Q0;
            const int q1 = Q1;
            delta0 = av_clip((((q0 - p0) << 2) + p1 - q1 + 4) >> 3, -tc, tc);
            if(!no_p)
                P0 = av_clip_pixel(p0 + delta0);
            if(!no_q)
                Q0 = av_clip_pixel(q0 - delta0);
            pix += ystride;
        }
    }
}

static void FUNC(hevc_h_loop_filter_chroma)(uint8_t *pix, ptrdiff_t stride, int *tc, uint8_t *_no_p, uint8_t *_no_q)
{
    FUNC(hevc_loop_filter_chroma)(pix, stride, sizeof(pixel), tc, _no_p, _no_q);
}

static void FUNC(hevc_v_loop_filter_chroma)(uint8_t *pix, ptrdiff_t stride, int *tc, uint8_t *_no_p, uint8_t *_no_q)
{
    FUNC(hevc_loop_filter_chroma)(pix, sizeof(pixel), stride, tc, _no_p, _no_q);
}

static void FUNC(hevc_h_loop_filter_luma)(uint8_t *pix, ptrdiff_t stride, int *_beta, int *_tc, uint8_t *_no_p, uint8_t *_no_q)
{
    FUNC(hevc_loop_filter_luma)(pix, stride, sizeof(pixel), _beta, _tc, _no_p, _no_q);
}

static void FUNC(hevc_v_loop_filter_luma)(uint8_t *pix, ptrdiff_t stride, int *_beta, int *_tc, uint8_t *_no_p, uint8_t *_no_q)
{
    FUNC(hevc_loop_filter_luma)(pix, sizeof(pixel), stride, _beta, _tc, _no_p, _no_q);
}

#undef P3
#undef P2
#undef P1
#undef P0
#undef Q0
#undef Q1
#undef Q2
#undef Q3

#undef TP3
#undef TP2
#undef TP1
#undef TP0
#undef TQ0
#undef TQ1
#undef TQ2
#undef TQ3


#ifdef SVC_EXTENSION
#define LumHor_FILTER(pel, coeff) \
(pel[0]*coeff[0] + pel[1]*coeff[1] + pel[2]*coeff[2] + coeff[3]*pel[3] + pel[4]*coeff[4] + pel[5]*coeff[5] + pel[6]*coeff[6] + pel[7]*coeff[7])

#define CroHor_FILTER(pel, coeff) \
(pel[0]*coeff[0] + pel[1]*coeff[1] + pel[2]*coeff[2] + pel[3]*coeff[3])

#define LumVer_FILTER(pel, coeff) \
(pel[0]*coeff[0] + pel[1]*coeff[1] + pel[2]*coeff[2] + pel[3]*coeff[3] + pel[4]*coeff[4] + pel[5]*coeff[5] + pel[6]*coeff[6] + pel[7]*coeff[7])

#define CroVer_FILTER(pel, coeff) \
(pel[0]*coeff[0] + pel[1]*coeff[1] + pel[2]*coeff[2] + pel[3]*coeff[3])

#define LumVer_FILTER1(pel, coeff, width) \
(pel[0]*coeff[0] + pel[width]*coeff[1] + pel[width*2]*coeff[2] + pel[width*3]*coeff[3] + pel[width*4]*coeff[4] + pel[width*5]*coeff[5] + pel[width*6]*coeff[6] + pel[width*7]*coeff[7])
// Define the function for up-sampling
#define CroVer_FILTER1(pel, coeff, widthEL) \
(pel[0]*coeff[0] + pel[widthEL]*coeff[1] + pel[widthEL*2]*coeff[2] + pel[widthEL*3]*coeff[3])


static void FUNC(upsample_base_layer_frame)(struct AVFrame *FrameEL, struct AVFrame *FrameBL, short *Buffer[3], const int32_t enabled_up_sample_filter_luma[16][8], const int32_t enabled_up_sample_filter_chroma[16][4], struct HEVCWindow *Enhscal, struct UpsamplInf *up_info)
{
    int i,j, k;
   
    int widthBL =  FrameBL->width;
    int heightBL = FrameBL->height;
    int strideBL = FrameBL->linesize[0];
    int widthEL =  FrameEL->width - Enhscal->left_offset - Enhscal->right_offset;
    int heightEL = FrameEL->height - Enhscal->top_offset - Enhscal->bottom_offset;
    int strideEL = FrameEL->linesize[0];
    
    pixel *srcBufY = (pixel*)FrameBL->data[0];
    pixel *dstBufY = (pixel*)FrameEL->data[0];
    short *tempBufY = Buffer[0];
    pixel *srcY;
    pixel *dstY;
    short *dstY1;
    short *srcY1;
    pixel *srcBufU = (pixel*)FrameBL->data[1];
    pixel *dstBufU = (pixel*)FrameEL->data[1];
    short *tempBufU = Buffer[1];
    pixel *srcU;
    pixel *dstU;
    short *dstU1;
    short *srcU1;
    
    pixel *srcBufV = (pixel*)FrameBL->data[2];
    pixel *dstBufV = (pixel*)FrameEL->data[2];
    short *tempBufV = Buffer[2];
    pixel *srcV;
    pixel *dstV;
    short *dstV1;
    short *srcV1;
    
    int refPos16 = 0;
    int phase    = 0;
    int refPos   = 0;
    int32_t* coeff = enabled_up_sample_filter_chroma[phase];
    widthEL   = FrameEL->width;  //pcUsPic->getWidth ();
    heightEL  = FrameEL->height; //pcUsPic->getHeight();
    
    widthBL   = FrameBL->width;
    heightBL  = FrameBL->height <= heightEL ? FrameBL->height:heightEL;  // min( FrameBL->height, heightEL);
    int leftStartL = Enhscal->left_offset;
    int rightEndL  = FrameEL->width - Enhscal->right_offset;
    int topStartL  = Enhscal->top_offset;
    int bottomEndL = FrameEL->height - Enhscal->bottom_offset;
    
    pixel buffer[8];
    for( i = 0; i < widthEL; i++ )	{
    	int x = av_clip_c(i, leftStartL, rightEndL);
        refPos16 = (((x - leftStartL)*up_info->scaleXLum + up_info->addXLum) >> 12);
        phase    = refPos16 & 15;
        refPos   = refPos16 >> 4;
        coeff = enabled_up_sample_filter_luma[phase];
        refPos -= ((NTAPS_LUMA>>1) - 1);
        srcY = srcBufY + refPos;
        dstY1 = tempBufY + i;
        if(refPos < 0)
            for( j = 0; j < heightBL ; j++ )	{
                
        		memset(buffer, srcY[-refPos], -refPos);
                memcpy(buffer-refPos, srcY-refPos, 8+refPos);
                *dstY1 = LumHor_FILTER(buffer, coeff);
                
                srcY += strideBL;
                dstY1 += widthEL;//strideEL;
            }else if(refPos+8 > widthBL )
                for( j = 0; j < heightBL ; j++ )	{
                    
            		memcpy(buffer, srcY, widthBL-refPos);
                    memset(buffer+(widthBL-refPos), srcY[widthBL-refPos-1], 8-(widthBL-refPos));
                    *dstY1 = LumHor_FILTER(buffer, coeff);
                    
                    srcY += strideBL;
                    dstY1 += widthEL;//strideEL;
                }else
                    for( j = 0; j < heightBL ; j++ )	{
                        
                        *dstY1 = LumHor_FILTER(srcY, coeff);
                        srcY += strideBL;
                        dstY1 += widthEL;//strideEL;
                    }
        
    }
    const int nShift = US_FILTER_PREC*2;
    int iOffset = 1 << (nShift - 1);
    short buffer1[8];
    for( j = 0; j < heightEL; j++ )	{
    	int y = av_clip_c(j, topStartL, bottomEndL-1);
    	refPos16 = ((( y - topStartL )*up_info->scaleYLum + up_info->addYLum) >> 12);
        phase    = refPos16 & 15;
        refPos   = refPos16 >> 4;
        coeff = enabled_up_sample_filter_luma[phase];
        refPos -= ((NTAPS_LUMA>>1) - 1);
        srcY1 = tempBufY + refPos *widthEL;
        dstY = dstBufY + j * strideEL;
        if(refPos < 0)
            for( i = 0; i < widthEL; i++ )	{
                
        		for(k= 0; k<-refPos ; k++)
        			buffer1[k] = srcY1[-refPos*widthEL]; //srcY1[(-refPos+k)*strideEL];
                for(k= 0; k<8+refPos ; k++)
                	buffer1[-refPos+k] = srcY1[(-refPos+k)*widthEL];
                *dstY = av_clip_pixel( (LumVer_FILTER(buffer1, coeff) + iOffset) >> (nShift));
                
                if( (i >= leftStartL) && (i <= rightEndL-2) )
                    srcY1++;
                dstY++;
            }else if(refPos+8 > heightBL )
                for( i = 0; i < widthEL; i++ )	{
                    
                    for(k= 0; k<heightBL-refPos ; k++)
                        buffer1[k] = srcY1[k*widthEL];
                    for(k= 0; k<8-(heightBL-refPos) ; k++)
                        buffer1[heightBL-refPos+k] = srcY1[(heightBL-refPos-1)*widthEL];
                    *dstY = av_clip_pixel( (LumVer_FILTER(buffer1, coeff) + iOffset) >> (nShift));
                    
                    if( (i >= leftStartL) && (i <= rightEndL-2) )
                        srcY1++;
                    dstY++;
                }else
                    for( i = 0; i < widthEL; i++ )	{
                        
                        *dstY = av_clip_pixel( (LumVer_FILTER1(srcY1, coeff, widthEL) + iOffset) >> (nShift));
                        
                        if( (i >= leftStartL) && (i <= rightEndL-2) )
                            srcY1++;
                        dstY++;
                    }
    }
    widthBL   = FrameBL->width;
    heightBL  = FrameBL->height;
    
    widthEL   = FrameEL->width - Enhscal->right_offset - Enhscal->left_offset;
    heightEL  = FrameEL->height - Enhscal->top_offset - Enhscal->bottom_offset;
    
    widthEL  >>= 1;
    heightEL >>= 1;
    widthBL  >>= 1;
    heightBL >>= 1;
    strideBL  = FrameBL->linesize[1];
    strideEL  = FrameEL->linesize[1];
    int leftStartC = Enhscal->left_offset>>1;
    int rightEndC  = (FrameEL->width>>1) - (Enhscal->right_offset>>1);
    int topStartC  = Enhscal->top_offset>>1;
    int bottomEndC = (FrameEL->height>>1) - (Enhscal->bottom_offset>>1);
    
    
    widthEL   = FrameEL->width >> 1;
    heightEL  = FrameEL->height >> 1;
    widthBL   = FrameBL->width >> 1;
    heightBL  = FrameBL->height > heightEL ? FrameBL->height:heightEL;
    
    
    heightBL >>= 1;
    
    //========== horizontal upsampling ===========
    for( i = 0; i < widthEL; i++ )	{
    	int x = av_clip_c(i, leftStartC, rightEndC - 1);
        refPos16 = (((x - leftStartC)*up_info->scaleXCr + up_info->addXCr) >> 12);
        phase    = refPos16 & 15;
        refPos   = refPos16 >> 4;
        coeff = enabled_up_sample_filter_chroma[phase];
        
        refPos -= ((NTAPS_CHROMA>>1) - 1);
        srcU = srcBufU + refPos; // -((NTAPS_CHROMA>>1) - 1);
        srcV = srcBufV + refPos; // -((NTAPS_CHROMA>>1) - 1);
        dstU1 = tempBufU + i;
        dstV1 = tempBufV + i;
        
        if(refPos < 0)
            for( j = 0; j < heightBL ; j++ )	{
                
        		memset(buffer, srcU[-refPos], -refPos);
                memcpy(buffer-refPos, srcU-refPos, 4+refPos);
                memset(buffer+4, srcV[-refPos], -refPos);
                memcpy(buffer-refPos+4, srcV-refPos, 4+refPos);
                
                *dstU1 = CroHor_FILTER(buffer, coeff);
                
                *dstV1 = CroHor_FILTER((buffer+4), coeff);
                
                
                srcU += strideBL;
                srcV += strideBL;
                dstU1 += widthEL;
                dstV1 += widthEL;
        	}else if(refPos+4 > widthBL )
                for( j = 0; j < heightBL ; j++ )	{
                    
                    memcpy(buffer, srcU, widthBL-refPos);
                    memset(buffer+(widthBL-refPos), srcU[widthBL-refPos-1], 4-(widthBL-refPos));
                    
                    memcpy(buffer+4, srcV, widthBL-refPos);
                    memset(buffer+(widthBL-refPos)+4, srcV[widthBL-refPos-1], 4-(widthBL-refPos));
                    
                    *dstU1 = CroHor_FILTER(buffer, coeff);
                    
                    *dstV1 = CroHor_FILTER((buffer+4), coeff);
                    
                	srcU += strideBL;
                    srcV += strideBL;
                    dstU1 += widthEL;
                    dstV1 += widthEL;
                }else
                    for( j = 0; j < heightBL ; j++ )	{
                        
                		*dstU1 = CroHor_FILTER(srcU, coeff);
                        
                		*dstV1 = CroHor_FILTER(srcV, coeff);
                        
                        
                        srcU += strideBL;
                        srcV += strideBL;
                        dstU1 += widthEL;
                        dstV1 += widthEL;
                	}
    }
    
    
    for( j = 0; j < heightEL; j++ )	{
        int y = av_clip_c(j, topStartC, bottomEndC - 1);
        refPos16 = (((y - topStartC)*up_info->scaleYCr + up_info->addYCr) >> 12) - 4;
        phase    = refPos16 & 15;
        refPos   = refPos16 >> 4;
        coeff = enabled_up_sample_filter_chroma[phase];
        refPos -= ((NTAPS_CHROMA>>1) - 1);
        srcU1 = tempBufU  + refPos *widthEL;
        srcV1 = tempBufV  + refPos *widthEL;
        dstU = dstBufU + j*strideEL;
        dstV = dstBufV + j*strideEL;
        if(refPos < 0)
            for( i = 0; i < widthEL; i++ )	{
                
                for(k= 0; k<-refPos ; k++){
                    buffer1[k] = srcU1[(-refPos)*widthEL];
                    buffer1[k+4] = srcV1[(-refPos)*widthEL];
                }
                for(k= 0; k<4+refPos ; k++){
                    buffer1[-refPos+k] = srcU1[(-refPos+k)*widthEL];
                    buffer1[-refPos+k+4] = srcV1[(-refPos+k)*widthEL];
                }
                *dstU = av_clip_pixel( (CroVer_FILTER(buffer1, coeff) + iOffset) >> (nShift));
                *dstV = av_clip_pixel( (CroVer_FILTER((buffer1+4), coeff) + iOffset) >> (nShift));
                
                if( (i >= leftStartC) && (i <= rightEndC-2) )	{
                    srcU1++;
                    srcV1++;
                }
                dstU++;
                dstV++;
            }else if(refPos+4 > heightBL )
                for( i = 0; i < widthEL; i++ )	{
                    
                    for(k= 0; k< heightBL-refPos ; k++) {
                        buffer1[k] = srcU1[k*widthEL];
                        buffer1[k+4] = srcV1[k*widthEL];
                    }
                    for(k= 0; k<4-(heightBL-refPos) ; k++) {
                        buffer1[heightBL-refPos+k] = srcU1[(heightBL-refPos-1)*widthEL];
                        buffer1[heightBL-refPos+k+4] = srcV1[(heightBL-refPos-1)*widthEL];
                    }
                    *dstU = av_clip_pixel( (CroVer_FILTER(buffer1, coeff) + iOffset) >> (nShift));
                    
                    
                    *dstV = av_clip_pixel( (CroVer_FILTER((buffer1+4), coeff) + iOffset) >> (nShift));
                    
                    if( (i >= leftStartC) && (i <= rightEndC-2) )	{
                        srcU1++;
                        srcV1++;
                    }
                    dstU++;
                    dstV++;
                }else
                    for( i = 0; i < widthEL; i++ )	{
                        *dstU = av_clip_pixel( (CroVer_FILTER1(srcU1, coeff, widthEL) + iOffset) >> (nShift));
                        
                        
                        *dstV = av_clip_pixel( (CroVer_FILTER1(srcV1, coeff, widthEL) + iOffset) >> (nShift));
                        
                        if( (i >= leftStartC) && (i <= rightEndC-2) )	{
                            srcU1++;
                            srcV1++;
                        }
                        dstU++;
                        dstV++;
                    }
    }
}

#undef LumHor_FILTER
#undef LumCro_FILTER
#undef LumVer_FILTER
#undef CroVer_FILTER
#endif


