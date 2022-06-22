// Copyright 2022 ETH Zurich, University of Bologna and Polytechnique Montreal.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: MohammadHossein AskariHemmat <m.h.askari.hemmat@gmail.com>

#ifndef BITSERIAL_MATMUL_H
#define BITSERIAL_MATMUL_H

#include <stdint.h>

void bitserial_matmul_init();
void bitserial_matmul_64(uint64_t* c, uint64_t* a, uint64_t* b, int aprec, int bprec);
int im2col_get_pixel(int *im, int height, int width, int channels, int row, int col, int channel, int pad);
void im2col_cpu(int* data_im, int channels,  int height,  int width, int ksize,  int stride, int pad, int* data_col);
void fill_cpu(int N, float ALPHA, float *X, int INCX);
void gemm(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc);
void conv2d(convolutional_layer l, network net);

#endif


#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "layer.h"

typedef layer convolutional_layer;
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, int batch_normalize, int binary, int xnor, int adam);

#endif
