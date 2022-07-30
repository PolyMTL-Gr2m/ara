#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>

#define TILE_SIZE 128
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)



void fconv2d_tensor64(double *o, double *i, double *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void fconv2d_tensor64_vec_4xC_1x1(double *o, double *i, double *f, int64_t R, int64_t C, int64_t W, int64_t F);
void fconv2d_tensor64_vec_6xC_3x3(double *o, double *i, double *f, int64_t R, int64_t C, int64_t W, int64_t F);
void fconv2d_tensor64_vec_6xC_5x5(double *o, double *i, double *f, int64_t R, int64_t C, int64_t W, int64_t F);
void fconv2d_tensor64_vec_4xC_7x7(double *o, double *i, double *f, int64_t R, int64_t C, int64_t W, int64_t F);
                        



#endif
