#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>


#define TILE_SIZE_WIDE 128
#define TILE_SIZE_WIDE_OUT (TILE_SIZE_WIDE - F + 1)

#define TILE_SIZE 256
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)


void fconv2d_tensor32(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void fconv2d_tensor32_vec_4xC_1x1(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F);
void fconv2d_tensor32_vec_6xC_3x3(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F);
void fconv2d_tensor32_vec_6xC_5x5(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F);
void fconv2d_tensor32_vec_4xC_7x7(float *o, float *i, float *f, int64_t R, int64_t C, int64_t W, int64_t F);


#endif