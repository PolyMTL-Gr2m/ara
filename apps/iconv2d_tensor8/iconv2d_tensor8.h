#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>

// WIDE MACRO

#define TILE_SIZE_WIDE 256
#define TILE_SIZE_WIDE_OUT (TILE_SIZE_WIDE - F + 1)

// SAME PRECISION MACRO

#define TILE_SIZE 1024
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)

#define TILE_SIZE_ 512
#define TILE_SIZE_OUT_ (TILE_SIZE - F + 1)
//FOR TEST ONLY
void iconv2d_tensor8_naive_wide(int32_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);


void iconv2d_tensor8(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void iconv2d_tensor8_vec_1xC_1x1(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor8_vec_4xC_3x3(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor8_vec_6xC_5x5(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor8_vec_4xC_7x7(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);

void iconv2d_tensor8_wide(int32_t *o,  int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void iconv2d_tensor8_vec_1xC_1x1_wide(int32_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor8_vec_8xC_3x3_wide(int32_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor8_vec_6xC_5x5_wide(int32_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor8_vec_6xC_7x7_wide(int32_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);


// WORK IN PROGRESS (NOT WORKING)

void iconv2d_tensor8_NHWC(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W,int64_t F, int64_t K);
void iconv2d_tensor8_filter_load_1x1(int8_t *f, int64_t W);
void iconv2d_tensor8_vec_4xW_1x1(int8_t *o, int8_t *i, int64_t C, int64_t W); 

void iconv2d_tensor8_NHWC(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W,int64_t F, int64_t K);
void iconv2d_tensor8_filter_load_3x3(int8_t *f, int64_t W);
void iconv2d_tensor8_vec_4xW_3x3(int8_t *o, int8_t *i, int64_t C, int64_t W); 


                        



#endif
