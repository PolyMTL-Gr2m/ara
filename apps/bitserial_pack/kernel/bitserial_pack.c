// Copyright 2022 ETH Zurich, University of Bologna and Polytechnique Montreal.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: MohammadHossein AskariHemmat <m.h.askari.hemmat@gmail.com>

#include "bitserial_pack.h"

void bitserial_pack_init() {
  asm volatile("vmv.v.i v0,  0");
  asm volatile("vmv.v.i v1,  0");
  asm volatile("vmv.v.i v2,  0");
}
// Assuming an array of 2bit precision data is packed into 8bit chars, this function,
// de-reference these chars into 64b words. Each call to VBPACK will consume 4 words
// (4 lanes) of 64 bits.
void bitserial_pack_64(uint64_t* mat_i, uint64_t* out, uint64_t len_mat_i) {
    uint64_t vl=0;
    uint64_t complete_words = len_mat_i/4;
    uint64_t residuals = len_mat_i%4;
    uint64_t elm_cnt = 0;
    // Make sure SEW is set to 64
    asm volatile("vsetvli zero, %0, e64, m1, ta, ma" ::"r"(4));
    // start with a fresh temp registers
    bitserial_pack_init();
    for (uint64_t n=0; n<4; n++){
        for (uint64_t i=0; i<4; i++){
            asm volatile("vle64.v v1, (%[A])" : : [A] "r" (mat_i));
            asm volatile(".byte 0x57, 0x81, 0x20, 0x0E\n" ::);
        }
        asm volatile("vse64.v v2, (%0)" : "+&r"(out));
        out+= 4;
    }
}