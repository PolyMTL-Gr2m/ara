#include <stdint.h>

#define TILE_SIZE 512
#define TILE_SIZE_3x3 510 // needs to be divisible by 3

// INTEGER

void imax_pool32(int32_t *o, int32_t *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);
void iavg_pool32(int32_t *o, int32_t *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);

void imax_pool_vec_1xC_2x2(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void imax_pool_vec_1xC_3x3(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);

void iavg_pool_vec_1xC_2x2(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void iavg_pool_vec_1xC_3x3(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
