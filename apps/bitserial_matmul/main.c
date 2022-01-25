// Copyright 2022 ETH Zurich, University of Bologna and Polytechnique Montreal.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: MohammadHossein AskariHemmat <m.h.askari.hemmat@gmail.com>

#include <stdint.h>
#include <string.h>

#include "printf.h"
#include "runtime.h"
#include "kernel/bitserial_matmul.h"

// Define Matrix dimensions:
// C = AB with A=[MxN], B=[NxP], C=[MxP]
#define M 64
#define N 64
#define P 64

uint64_t a[M * N] __attribute__((aligned(32 * NR_LANES), section(".l2")));
uint64_t b[N * P] __attribute__((aligned(32 * NR_LANES), section(".l2")));
uint64_t c[M * P] __attribute__((aligned(32 * NR_LANES), section(".l2")));

// Initialize the matrices
void init_matrix(uint64_t *matrix, uint64_t num_rows, uint64_t num_columns) {
  for (uint64_t i = 0; i < num_rows; ++i) {
    for (uint64_t j = 0; j < num_columns; ++j) {
      matrix[i * num_columns + j] = i;
    }
  }
}


int main(void) {
    const int s = 64;
    const int prec = 2;
    printf("\n");
    printf("------------------------------------------------------------\n");
    printf("Calculating a bit-serialized (%d x %d) x (%d x %d) matrix multiplication with 2 bit precision...\n", s,
           s, s, s);
    printf("------------------------------------------------------------\n");
    printf("\n");
    // Initialize Matrices
    printf("Initializing matrices...\n");
    // init_matrix(a, s, s);
    // init_matrix(b, s, s);
    // Matrices are initialized --> Start calculating
    printf("Calculating bitserial_matmul_64...\n");
    start_timer();
    bitserial_matmul_64(c, a, b, 1, prec);
    stop_timer();
    // Metrics
    int64_t runtime = get_timer();
    float performance = 2.0 * s * s / runtime;
    float utilization = 100 * performance / (2.0 * NR_LANES);

    printf("The execution took %d cycles.\n", runtime);
    printf("The performance is %f OP/cycle (%f%% utilization).\n", performance,
           utilization);

}
