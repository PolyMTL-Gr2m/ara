#include <stdint.h>

#define TILE_SIZE 256
#define TILE_SIZE_3x3 255 // needs to be divisible by 3

// float

void fmax_pool32(float *o, float *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);
void favg_pool32(float *o, float *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);

void fmax_pool_vec_1xC_2x2(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void fmax_pool_vec_1xC_3x3(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);

void favg_pool_vec_1xC_2x2(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void favg_pool_vec_1xC_3x3(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
