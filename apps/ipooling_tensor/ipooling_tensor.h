#include <stdint.h>

// SAME PRECISION MACRO

#define TILE_SIZE 1024
#define TILE_SIZE_3x3 1023 // needs to be divisible by 3

// INTEGER

// 8b

void imax_pool8(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W, int64_t F);
void iavg_pool8(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W, int64_t F);

void imax_pool_vec_1xC_2x2(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void imax_pool_vec_1xC_3x3(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);

void iavg_pool_vec_1xC_2x2(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);
void iavg_pool_vec_1xC_3x3(int8_t *o, int8_t *i, int64_t R, int64_t C, int64_t W, int64_t F, int64_t stride);

/*
// 16b

void imax_pool16(int16_t *o, int16_t *i, int64_t R, int64_t C, int64_t W, int64_t F);
void iavg_pool16(int16_t *o, int16_t *i, int64_t R, int64_t C, int64_t W, int64_t F);

// 32b

void imax_pool32(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F);
void iavg_pool32(int32_t *o, int32_t *i, int64_t R, int64_t C, int64_t W, int64_t F);

// 64b

void imax_pool64(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W, int64_t F);
void iavg_pool64(int64_t *o, int64_t *i, int64_t R, int64_t C, int64_t W, int64_t F);


// FLOAT

// 32b

void fmax_pool32(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F);
void favg_pool32(float *o, float *i, int64_t R, int64_t C, int64_t W, int64_t F);

// 64b

void fmax_pool64(double *o, double *i, int64_t R, int64_t C, int64_t W, int64_t F);
void favg_pool64(double *o, double *i, int64_t R, int64_t C, int64_t W, int64_t F);
*/

