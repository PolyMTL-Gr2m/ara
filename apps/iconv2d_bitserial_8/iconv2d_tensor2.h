#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif

// WIDE MACRO

#define TILE_SIZE_WIDE 256
#define TILE_SIZE_WIDE_OUT (TILE_SIZE_WIDE - F + 1)

// SAME PRECISION MACRO

#define TILE_SIZE 1024
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)

#define TILE_SIZE_ 512
#define TILE_SIZE_OUT_ (TILE_SIZE - F + 1)


void conv2d_prec2(int8_t * i_ptr, int8_t *f_ptr, int8_t *o_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);

void conv2d_prec1(int8_t * i_ptr, int8_t *f_ptr, int8_t *o_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);



// BIT PACKING FUNCTIONS (will have its own folder as lib at some point)

//void bitpack_2b(uint8_t* tensor, uint8_t* packed_data, uint64_t DATA_WIDTH, uint64_t len);

void vbitpack(uint8_t* tensor, uint64_t size);
void bitpack_filter(uint8_t* tensor, uint8_t* packed_data, uint64_t len);


void vbitpack_1(uint8_t* tensor, uint64_t size);
void bitpack_filter_1(uint8_t* tensor, uint8_t* packed_data, uint64_t len);

                        



#endif
