#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>

#ifndef SPIKE
#include "printf.h"
#include "runtime.h"
#endif


void conv2d(int64_t * i_ptr, int64_t *f_ptr, int64_t *o_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);

void conv2d_prec1(int64_t * i_ptr, int64_t *f_ptr, int64_t *o_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);

void conv2d_prec1_test(int64_t * i_ptr, int64_t *f_ptr, int64_t *o_ptr, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);



// BIT PACKING FUNCTIONS (will have its own folder as lib at some point)

//void bitpack_2b(uint8_t* tensor, uint8_t* packed_data, uint64_t DATA_WIDTH, uint64_t len);

void vbitpack(uint64_t* tensor, uint64_t size);
void bitpack_filter(uint64_t* tensor, uint64_t* packed_data, uint64_t len);


void vbitpack_1(uint64_t* tensor, uint64_t size);
void bitpack_filter_1(uint64_t* tensor, uint64_t* packed_data, uint64_t len);

                        



#endif
