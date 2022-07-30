#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>


#define TILE_SIZE 512
#define TILE_SIZE_OUT (TILE_SIZE - F + 1)


void iReLu_tensor8(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W);

void iReLu_tensor8_vec_8xC(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W);



#endif
