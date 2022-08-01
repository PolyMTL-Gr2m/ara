#include <stdint.h>

// SAME PRECISION MACRO

#define TILE_SIZE 256
#define TILE_SIZE_3x3 126 // needs to be divisible by 3

// INTEGER

void fmax_pool64(double *o, double *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);
void favg_pool64(double *o, double *i, int64_t H_in, int64_t W_in, int64_t C_in, int64_t F);

void fmax_pool_vec_1xC_2x2(double *o, double *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void fmax_pool_vec_1xC_3x3(double *o, double *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);

void favg_pool_vec_1xC_2x2(double *o, double *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void favg_pool_vec_1xC_3x3(double *o, double *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);

