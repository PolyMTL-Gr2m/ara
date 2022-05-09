#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>


void iconv2d_tensor8(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F, int64_t K);

void iconv2d_tensor8_vec_16xC_1x1(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor8_vec_16xC_3x3(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);
void iconv2d_tensor8_vec_16xC_5x5(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W, int64_t F);


void iconv2d_tensor8_1x1_NWHC(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W,int64_t F, int64_t K);
void iconv2d_tensor8_filter_load_1x1(int8_t *f, int64_t W);
void iconv2d_tensor8_vec_4xW_1x1(int8_t *o, int8_t *i, int64_t C, int64_t W); 

void iconv2d_tensor8_3x3_NWHC(int8_t *o, int8_t *i, int8_t *f, int64_t R, int64_t C, int64_t W,int64_t F, int64_t K);
void iconv2d_tensor8_filter_load_3x3(int8_t *f, int64_t W);
void iconv2d_tensor8_vec_4xW_3x3(int8_t *o, int8_t *i, int64_t C, int64_t W); 



                        


/*void iconv2d_tensor8_5x5(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C,
                int64_t F);
void iconv2d_tensor_vec_4xC_slice_init_5x5(int64_t *o, int64_t C);
void iconv2d_tensor_vec_4xC_slice_preload_5x5(int64_t *i, int64_t C, int64_t F);
void iconv2d_tensor_vec_4xC_slice_move_5x5(int64_t C, int64_t F);
void iconv2d_tensor_vec_4xC_5x5(int64_t *o, int64_t *i, int64_t *f, int64_t C,
                        int64_t F);

void iconv2d_tensor_7x7(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C,
                int64_t F);
void iconv2d_tensor_vec_4xC_slice_init_7x7(int64_t *o, int64_t C);
void iconv2d_tensor_vec_4xC_slice_preload_7x7(int64_t *i, int64_t C, int64_t F);
void iconv2d_tensor_vec_4xC_slice_move_7x7(int64_t C, int64_t F);
void iconv2d_tensor_vec_4xC_7x7(int64_t *o, int64_t *i, int64_t *f, int64_t C,
                        int64_t F);*/

#endif
