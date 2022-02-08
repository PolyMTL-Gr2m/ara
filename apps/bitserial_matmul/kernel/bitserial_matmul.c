// Copyright 2022 ETH Zurich, University of Bologna and Polytechnique Montreal.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: MohammadHossein AskariHemmat <m.h.askari.hemmat@gmail.com>

#include "bitserial_matmul.h"

void bitserial_matmul_init() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v4,  0");
  asm volatile("vmv.v.i v8,  0");
  asm volatile("vmv.v.i v12, 0");
  asm volatile("vmv.v.i v16, 0");
}

// given the input arrays with bit precision prec in bitpack format, this function
// calculates the Matmul of the two Matrix and returns the resulting Matrix.
// NOTE: a and b must have the same length. 
// NOTE: a and b must be in bit packed format, the retuning matrix is in normal format.
// c[64x64] = a[64]*b[64]
void bitserial_matmul_64(uint64_t* c, uint64_t* a, uint64_t* b, int aprec, int bprec) {
    uint64_t vl=0;
    // Original pointers
    const uint64_t *a_ = a;
    const uint64_t *b_ = b;
    uint64_t elen = 64;
    asm volatile("vsetvli %0, %1, e64, m1, ta, ma \n" : "+r" (vl) : "r" (elen));
    // start with a fresh temp registers
    bitserial_matmul_init();
    for (int row=0; row<64; row++){
        // The following loop will compute one row of a 64 element output
        for (int i=0; i<aprec; i++){
            for (int j=0; j<bprec; j++){
                // load a
                asm volatile("vle64.v v0, (%[A])" : : [A] "r" (a));
                // load b
                asm volatile("vle64.v v1, (%[A])" : : [A] "r" (b));
                // broadcast one row to the vector
                asm volatile("vrgather.vx v4, v1, %0" : : "r" (row));
                // v4 = v0 & v4
                asm volatile("vand.vv v4, v0, v1" ::);
                // v8 = vpopcnt(v4) 
                __asm__ volatile(".byte 0x57, 0x04, 0x22, 0x06\n" ::);
                // partial sum in v12
                asm volatile("vadd.vv v12, v12, v8" ::);
                a += vl;
                b += vl;
            }
        }
        // Done with the first row of b. Roll back a to the begining
        a = a_;
        b = b_;
        // And save result (first row) to output
        asm volatile("vse64.v v12, (%0);" ::"r"(c));
        c += vl;
        // Reset result vector register
        asm volatile("vmv.v.i v12, 0");
    }
}