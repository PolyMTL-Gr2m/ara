#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>



#define TILE_SIZE_WIDE 256
#define TILE_SIZE_WIDE_OUT (TILE_SIZE_WIDE - F + 1)

#define TILE_SIZE_1x1_WIDE 1024

#define TILE_SIZE 512
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)

#define TILE_SIZE_1x1 2048


void iconv2d_tensor16(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void iconv2d_tensor16_vec_1xC_1x1(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor16_vec_6xC_3x3(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor16_vec_6xC_5x5(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor16_vec_4xC_7x7(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);

void iconv2d_tensor16_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void iconv2d_tensor16_vec_1xC_1x1_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor16_vec_4xC_3x3_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor16_vec_8xC_5x5_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor16_vec_6xC_7x7_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
                        



#endif
