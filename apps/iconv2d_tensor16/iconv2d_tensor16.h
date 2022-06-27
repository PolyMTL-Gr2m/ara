#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>

#define next_plane_2  (R + F - 2)*(C + F - 1) << 1
#define next_plane_4  (R + F - 4)*(C + F - 1) << 1
#define next_plane_6  (R + F - 6)*(C + F - 1) << 1
#define next_plane_8  (R + F - 8)*(C + F - 1) << 1
#define next_plane_10 (R + F - 10)*(C + F - 1) << 1
#define next_plane_16 (R + F - 16)*(C + F - 1) << 1

#define block_size_1x1 8
#define block_size_3x3 8
#define block_size_5x5 4
#define block_size_7x7 4


#define TILE_SIZE 256
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)


void iconv2d_tensor(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

/*void iconv2d_tensor_vec_16xC_1x1(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor_vec_16xC_3x3(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor_vec_16xC_5x5(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);

void iconv2d_tensor_1x1_NHWC(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W,int64_t F, int64_t K);
void iconv2d_tensor_filter_load_1x1(int16_t *f, int64_t W);
void iconv2d_tensor_vec_4xW_1x1(int16_t *o, int16_t *i, int64_t C, int64_t W); 

void iconv2d_tensor_3x3_NHWC(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W,int64_t F, int64_t K);
void iconv2d_tensor_filter_load_3x3(int16_t *f, int64_t W);
void iconv2d_tensor_vec_4xW_3x3(int16_t *o, int16_t *i, int64_t C, int64_t W); */

void iconv2d_tensor16_vec_8xC_1x1_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor16_vec_8xC_3x3_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor16_vec_4xC_5x5_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor16_vec_2xC_7x7_wide(int32_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
                        



#endif
