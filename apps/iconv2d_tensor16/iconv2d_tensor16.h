#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>

void iconv2d_tensor16_3x3(int16_t *o, int16_t *i, int16_t *f, int64_t R, int64_t C, int64_t W,
                int64_t F, int64_t K);
//void iconv2d_vec_4xC_slice_init_3x3(int64_t *o, int64_t C);
void iconv2d_tensor16_vec_4xC_slice_preload_3x3(int16_t *i, int64_t C, int64_t F);
//void iconv2d_vec_4xC_slice_move_3x3(int64_t C, int64_t F);
void iconv2d_tensor16_vec_4xC_3x3(int16_t *o, int16_t *i, int16_t *f, int64_t C, int64_t W,
                        int64_t F);

/*void iconv2d_tensor16_5x5(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C,
                int64_t F);
void iconv2d_tensor16_vec_4xC_slice_init_5x5(int64_t *o, int64_t C);
void iconv2d_tensor16_vec_4xC_slice_preload_5x5(int64_t *i, int64_t C, int64_t F);
void iconv2d_tensor16_vec_4xC_slice_move_5x5(int64_t C, int64_t F);
void iconv2d_tensor16_vec_4xC_5x5(int64_t *o, int64_t *i, int64_t *f, int64_t C,
                        int64_t F);

void iconv2d_tensor16_7x7(int64_t *o, int64_t *i, int64_t *f, int64_t R, int64_t C,
                int64_t F);
void iconv2d_tensor16_vec_4xC_slice_init_7x7(int64_t *o, int64_t C);
void iconv2d_tensor16_vec_4xC_slice_preload_7x7(int64_t *i, int64_t C, int64_t F);
void iconv2d_tensor16_vec_4xC_slice_move_7x7(int64_t C, int64_t F);
void iconv2d_tensor16_vec_4xC_7x7(int64_t *o, int64_t *i, int64_t *f, int64_t C,
                        int64_t F);*/

#endif
