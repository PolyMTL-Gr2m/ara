// Copyright 2022 ETH Zurich, University of Bologna and Polytechnique Montreal.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: MohammadHossein AskariHemmat <m.h.askari.hemmat@gmail.com>

#include <stdint.h>
#include <string.h>

#include "printf.h"
#include "runtime.h"
#include <inttypes.h>
#include "kernel/bitserial_pack.h"

// Define Matrix dimensions:
// C = AB with A=[MxN], B=[NxP], C=[MxP]
#define M 64
#define N 64
#define P 64

uint64_t a[M * N] __attribute__((aligned(32 * NR_LANES), section(".l2")));
uint64_t b[N * P] __attribute__((aligned(32 * NR_LANES), section(".l2")));
uint64_t c[M * P] __attribute__((aligned(32 * NR_LANES), section(".l2")));

// Initialize the matrices
void init_matrix(uint64_t *matrix, uint64_t num_rows, uint64_t num_columns, int mask) {
  for (uint64_t i = 0; i < num_rows; ++i) {
    for (uint64_t j = 0; j < num_columns; ++j) {
      if  (mask == 0) matrix[i * num_columns + j] = 0x7555755575557555;
      else            matrix[i * num_columns + j] = 0xAAAAAAAAAAAAAAAA;
    }
  }
}


int main(void) {
    const int s = 4;
    // Initialize Matrices
    printf("Initializing matrices...\n");
    init_matrix(a, s, s, 0);
    init_matrix(b, s, s, 1);
    // init_matrix(b, s, s);
    // Matrices are initialized --> Start calculating
    //for (int i=0; i<s*s; i++){
    //    printf("[%4d]:  0x%" PRIx64 "\n", i, a[i]);
    //}
    //printf("Calculating bitserial_pack_64...\n");
    uint64_t* c_ptr = c;
    asm volatile("vsetvli zero, %0, e64, m1, ta, ma" ::"r"(4));
    asm volatile("vle64.v v1, (%[A])" : : [A] "r" (&a[0]));
    asm volatile("vle64.v v2, (%[A])" : : [A] "r" (&b[0]));
    start_timer();
    // bitserial_pack_64(a, c, s*s);
    asm volatile(".byte 0x57, 0x84, 0x20, 0x06\n" ::);
    stop_timer();
    int64_t runtime = get_timer();
    asm volatile("vse64.v v8, (%0)" : "+&r"(c_ptr));
    //printf("Results...\n");
    // for (int i=0; i<s*s; i++){
    //    printf("[%4d]:  0x%" PRIx64 "\n", i, c[i]);
    // }
    printf("The execution took %d cycles!\n", runtime);
    printf("result: 0x%" PRIx64 "\n", c[0]);
    // printf("The performance is %f OP/cycle (%f%% utilization).\n", performance,
    //        utilization);

}
