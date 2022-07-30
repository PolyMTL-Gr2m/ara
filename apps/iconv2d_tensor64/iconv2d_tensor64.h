#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>

#define TILE_SIZE 128
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)



void iconv2d_tensor64(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void iconv2d_tensor64_vec_4xC_1x1(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor64_vec_6xC_3x3(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor64_vec_6xC_5x5(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor64_vec_4xC_7x7(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
                        



#endif
