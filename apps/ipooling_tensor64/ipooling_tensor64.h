#include <stdint.h>

// SAME PRECISION MACRO

#define TILE_SIZE 256
#define TILE_SIZE_3x3 126 // needs to be divisible by 3

// INTEGER

// 8b

void imax_pool64(int64_t *o, int64_t *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);
void iavg_pool64(int64_t *o, int64_t *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);

void imax_pool_vec_1xC_2x2(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void imax_pool_vec_1xC_3x3(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);

void iavg_pool_vec_1xC_2x2(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void iavg_pool_vec_1xC_3x3(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);

