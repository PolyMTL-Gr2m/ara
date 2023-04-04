// Author : Theo Dupuis
// GR2M - 2022
// Polytechnique Montreal

#ifndef UP_CONV2D_H
#define UP_CONV2D_H

#include <stdint.h>

#include "runtime.h"

#ifndef SPIKE
#include "printf.h"
#endif


//#define SPARQ

// MACRO to specify the max size of vector registers (in this code, we consider VLEN = 4096)

// VLEN : length of register for EEW = 8b
#define VLEN_M2  1024
#define VLEN_M1  512
#define VLEN_MF2 256
#define VLEN_MF4 128

#define VLEN_M2_OUT  (VLEN_M2  - F + 1)
#define VLEN_M1_OUT  (VLEN_M1  - F + 1)
#define VLEN_MF2_OUT (VLEN_MF2 - F + 1)
#define VLEN_MF4_OUT (VLEN_MF4 - F + 1)


////////////////////////
// Selection function //
////////////////////////

void ulppack_conv2d(int16_t *o, int8_t *i, int8_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out, int64_t precA, int64_t precW);

// NO ACCELERATION VANILLA RISC-V "V"

void ulppack_conv2d_vec_7x7_A1W1(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ulppack_conv2d_vec_7x7_A2W2(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ulppack_conv2d_vec_7x7_A3W3(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);

// ACCELERATED WITH CUSTOM MUL SHIFT

void ulppack_conv2d_vec8_7x7(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ulppack_conv2d_vec16_7x7(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);


// WIP

void ulppack_conv2d_vec8_3x3(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ulppack_conv2d_vec16_3x3(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);


void ulppack_conv2d_vec_1x1_A1W1(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ulppack_conv2d_vec_1x1_A2W2(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);

void ulppack_conv2d_vec_3x3_A1W1(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ulppack_conv2d_vec_3x3_A2W2(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);
void ulppack_conv2d_vec_3x3_A3W3(int16_t * o_ptr, int8_t *i_ptr, int8_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out);





#endif
