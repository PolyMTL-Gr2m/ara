// Copyright 2021 ETH Zurich, University of Bologna and Polytechnique Montreal.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: MohammadHossein AskariHemmat
// <mohammadhossein.askari-hemmat@polymtl.ca>

#include "vector_macros.h"

void TEST_CASE1() {
  int8_t vrf1[16];
  int8_t vrf2[16];
  int8_t vrf3[16];

  VSET(16, e8, m1);
  VLOAD_8(v1, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8);
  VLOAD_8(v2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  VLOAD_8(v3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  // vpopcnt v1,v2,v3 --> v2 = popcnt(v1)
  __asm__ volatile(".byte 0xD7, 0x81, 0x20, 0x06\n" ::);

  VCMP_U8(1, v3, 1, 1, 2, 1, 2, 2, 3, 1, 1, 1, 2, 1, 2, 2, 3, 1);
}

int main(void) {
  INIT_CHECK();
  enable_vec();

  TEST_CASE1();

  EXIT_CHECK();
}
