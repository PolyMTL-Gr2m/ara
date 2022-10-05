#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>

#include "runtime.h"

#ifndef SPIKE
#include "printf.h"
#endif

#define TILE_SIZE 128
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)

#define TILE_SIZE_P3 128
#define TILE_SIZE_P3_OUT (TILE_SIZE_P3 - F + 1)

void ibsconv2d_tensor64_3x3(int64_t *o, int64_t *i, int64_t *f, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F, int64_t C_out, int64_t precA, int64_t precW);



void ibsconv2d64_W1_A1_vec_3x3(int64_t * o_ptr, int64_t *i_ptr, int64_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);
void ibsconv2d64_W2_A1_vec_3x3(int64_t * o_ptr, int64_t *i_ptr, int64_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);
void ibsconv2d64_W1_A2_vec_3x3(int64_t * o_ptr, int64_t *i_ptr, int64_t *f_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);
void ibsconv2d64_W2_A2_vec_3x3(int64_t * i_ptr, int64_t *f_ptr, int64_t *o_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);



// BIT PACKING FUNCTIONS (will have its own folder as lib at some point)

void bitpack64_vec_1_to_64(int64_t * tensor, uint64_t size);
void bitpack_filter64_vec_1_to_64(int64_t* tensor, int64_t* packed_data, uint64_t len);


void bitpack64_vec_2_to_64(int64_t * tensor, uint64_t size);
void bitpack_filter64_vec_2_to_64(int64_t * tensor, int64_t * packed_data, uint64_t len);
                        



#endif
