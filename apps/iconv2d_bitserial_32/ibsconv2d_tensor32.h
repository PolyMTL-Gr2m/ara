// Author : Theo Dupuis
// GR2M - 2022
// Polytechnique Montreal

#ifndef BSCONV2D_H
#define BSCONV2D_H

#include <stdint.h>

#include "runtime.h"

#ifndef SPIKE
#include "printf.h"
#endif

#define TILE_SIZE 256
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)

// MAX SIZE OF VREG FOR A1

#define TILE_SIZE_A1_3x3 128
#define TILE_SIZE_A1_3x3_OUT (TILE_SIZE_A1_3x3 - F + 1)

#define TILE_SIZE_A1_5x5 128
#define TILE_SIZE_A1_5x5_OUT (TILE_SIZE_A1_5x5 - F + 1)

#define TILE_SIZE_A1_7x7 128
#define TILE_SIZE_A1_7x7_OUT (TILE_SIZE_A1_7x7 - F + 1)

//#define PERF
//#define PERF_VHSACC

#ifdef PERF
	extern uint64_t runtime_cv;
	extern uint64_t runtime_bp;
#endif

void ibsconv2d_tensor32_3x3(int32_t *o, int8_t *i, int8_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out, int64_t precA, int64_t precW);

void ibsconv2d32_W1_A1_vec_3x3(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ibsconv2d32_W1_A1_vec_5x5(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ibsconv2d32_W1_A1_vec_7x7(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);

void ibsconv2d32_W2_A1_vec_3x3(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ibsconv2d32_W2_A1_vec_5x5(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ibsconv2d32_W2_A1_vec_7x7(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);

void ibsconv2d32_W2_A2_vec_3x3(int32_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);




void bitpack32_vec_1_to_32_2H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in);
void bitpack32_vec_1_to_32_4H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in);
void bitpack32_vec_1_to_32_6H(int8_t * tensor, uint64_t len, uint64_t C_in, uint64_t W_in);
void bitpack_filter32_vec_1_to_32(int8_t * tensor, int32_t* packed_data, uint64_t len, uint64_t C_in);

void bitpack32_vec_2_to_32_2H(int8_t * tensor, uint64_t len, uint64_t C_in);
void bitpack_filter32_vec_2_to_32(int8_t * tensor, int32_t* packed_data, uint64_t len, uint64_t C_in);



                        



#endif
