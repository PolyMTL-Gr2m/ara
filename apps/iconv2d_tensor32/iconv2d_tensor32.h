#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>


#define TILE_SIZE_WIDE 128
#define TILE_SIZE_WIDE_OUT (TILE_SIZE_WIDE - F + 1)

#define TILE_SIZE 256
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)


void iconv2d_tensor32(int32_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void iconv2d_tensor32_vec_4xC_1x1(int32_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor32_vec_6xC_3x3(int32_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor32_vec_6xC_5x5(int32_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor32_vec_4xC_7x7(int32_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F);

void iconv2d_tensor32_wide(int64_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void iconv2d_tensor32_vec_4xC_1x1_wide(int64_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor32_vec_4xC_3x3_wide(int64_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor32_vec_8xC_5x5_wide(int64_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor32_vec_6xC_7x7_wide(int64_t *o, int32_t *i, int32_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
                        



#endif
